from __future__ import annotations
import logging
import uuid
from typing import Any, Dict, List, Literal

from autogen_core import CancellationToken, Component, Image
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage


import array
import functools
import hashlib
import json
import logging
import os
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
    Literal
)

from pydantic import BaseModel, Field

from numpy.typing import NDArray

if TYPE_CHECKING:
    from oracledb import Connection

import numpy as np


logger = logging.getLogger(__name__)
log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

try:
    import oracledb
except ImportError as e:
    raise ImportError(
        "Unable to import oracledb, please install with "
        "`pip install -U oracledb`."
    ) from e


# Define a type variable that can be any kind of function
T = TypeVar("T", bound=Callable[..., Any])


class FilterCondition(TypedDict):
    key: str
    oper: str
    value: str


class FilterGroup(TypedDict, total=False):
    _and: Optional[List[Union["FilterCondition", "FilterGroup"]]]
    _or: Optional[List[Union["FilterCondition", "FilterGroup"]]]


def _convert_oper_to_sql(oper: str) -> str:
    oper_map = {"EQ": "==", "GT": ">", "LT": "<", "GTE": ">=", "LTE": "<="}
    if oper not in oper_map:
        raise ValueError("Filter operation {} not supported".format(oper))
    return oper_map.get(oper, "==")


def _generate_condition(condition: FilterCondition) -> str:
    key = condition["key"]
    oper = _convert_oper_to_sql(condition["oper"])
    value = condition["value"]
    if isinstance(value, str):
        value = f'"{value}"'
    return f"JSON_EXISTS(metadata, '$.{key}?(@ {oper} {value})')"


def _generate_where_clause(db_filter: Union[FilterCondition, FilterGroup]) -> str:
    if "key" in db_filter:  # Identify as FilterCondition
        return _generate_condition(cast(FilterCondition, db_filter))

    if "_and" in db_filter and db_filter["_and"] is not None:
        and_conditions = [
            _generate_where_clause(cond)
            for cond in db_filter["_and"]
            if isinstance(cond, dict)
        ]
        return "(" + " AND ".join(and_conditions) + ")"

    if "_or" in db_filter and db_filter["_or"] is not None:
        or_conditions = [
            _generate_where_clause(cond)
            for cond in db_filter["_or"]
            if isinstance(cond, dict)
        ]
        return "(" + " OR ".join(or_conditions) + ")"

    raise ValueError(f"Invalid filter structure: {db_filter}")

def _handle_exceptions(func: T) -> T:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except RuntimeError as db_err:
            # Handle a known type of error (e.g., DB-related) specifically
            logger.exception("DB-related error occurred.")
            raise RuntimeError(
                "Failed due to a DB issue: {}".format(db_err)
            ) from db_err
        except ValueError as val_err:
            # Handle another known type of error specifically
            logger.exception("Validation error.")
            raise ValueError("Validation failed: {}".format(val_err)) from val_err
        except Exception as e:
            # Generic handler for all other exceptions
            logger.exception("An unexpected error occurred: {}".format(e))
            raise RuntimeError("Unexpected error: {}".format(e)) from e

    return cast(T, wrapper)


async def _get_connection(client: Any) -> Connection | None:
    # check if AsyncConnectionPool exists
    connection_pool_class = getattr(oracledb, "AsyncConnectionPool", None)

    if isinstance(client, oracledb.AsyncConnection):
        return client
    elif connection_pool_class and isinstance(client, connection_pool_class):
        return await client.acquire()
    
    valid_types = "oracledb.AsyncConnection"
    if connection_pool_class:
        valid_types += " or oracledb.AsyncConnectionPool"
    raise TypeError(
        f"Expected client of type {valid_types}, got {type(client).__name__}"
    )

def _validate_version(connection: Union[oracledb.AsyncConnection, oracledb.AsyncConnectionPool]):
    # Check python client driver version 2.2.0
    if _compare_version(oracledb.__version__, "2.2.0"):
        raise Exception(
            f"Oracle DB python client driver version {oracledb.__version__} not supported, must be >=2.2.0"
        )

    # Check client version if thick 23ai
    if not (hasattr(connection, "thin") and connection.thin):

        if  _compare_version(".".join(map(str, oracledb.clientversion())), "23.4"):
            raise Exception(
                "Oracle DB python thick mode only supports client libraries "
                "greater than or equal to 23.4"
            )


def _compare_version(version: str, target_version: str) -> bool:
    # Split both version strings into parts
    version_parts = [int(part) for part in version.split(".")]
    target_parts = [int(part) for part in target_version.split(".")]

    # Compare each part
    for v, t in zip(version_parts, target_parts):
        if v < t:
            return True  # Current version is less
        elif v > t:
            return False  # Current version is greater

    # If all parts equal so far, check if version has fewer parts than target_version
    return len(version_parts) < len(target_parts)

async def _table_exists(connection: Union[oracledb.AsyncConnection, oracledb.AsyncConnectionPool], table_name: str) -> bool:

    try:
        with connection.cursor() as cursor:
            await cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            _ = await cursor.fetchone()
            return True
    except oracledb.DatabaseError as ex:
        err_obj = ex.args
        if err_obj[0].code == 942:
            return False
        raise


@_handle_exceptions
async def _get_clob_value(result: Any) -> str:
    clob_value = ""
    if result:
        if isinstance(result, oracledb.AsyncLOB):
            raw_data = await result.read()
            if isinstance(raw_data, bytes):
                clob_value = raw_data.decode(
                    "utf-8"
                )  # Specify the correct encoding
            else:
                clob_value = raw_data
        elif isinstance(result, str):
            clob_value = result
        else:
            raise Exception("Unexpected type:", type(result))
    return clob_value

@_handle_exceptions
async def _create_table(connection: Union[oracledb.AsyncConnection, oracledb.AsyncConnectionPool], table_name: str, modality: str, embedding_dim: int) -> None:

    cols_dict = {
        "id": "RAW(16) DEFAULT SYS_GUID() PRIMARY KEY",
        "model_input": "CLOB" if modality == "TEXT" else "BLOB",
        "metadata": "JSON",
        "embedding": f"vector({embedding_dim}, FLOAT32)",
    }

    table_exists = await _table_exists(connection, table_name)

    if not table_exists:
        with connection.cursor() as cursor:
            ddl_body = ", ".join(
                f"{col_name} {col_type}" for col_name, col_type in cols_dict.items()
            )
            ddl = f"CREATE TABLE {table_name} ({ddl_body})"
            await cursor.execute(ddl)

        logger.info("Table created successfully...")
    else:
        logger.info("Table already exists...")

@_handle_exceptions
async def _index_exists(connection: Connection, index_name: str) -> bool:
    # Check if the index exists
    query = """
        SELECT index_name 
        FROM all_indexes 
        WHERE upper(index_name) = upper(:idx_name)
        """

    with connection.cursor() as cursor:
        # Execute the query
        await cursor.execute(query, idx_name=index_name.upper())
        result = await cursor.fetchone()

    return result is not None

def _get_index_name(base_name: str) -> str:
    unique_id = str(uuid.uuid4()).replace("-", "")
    return f"{base_name}_{unique_id}"

@_handle_exceptions
async def _create_hnsw_index(
    connection: Connection,
    table_name: str,
    distance_strategy: DistanceStrategy,
    params: Optional[dict[str, Any]] = None,
) -> None:
    defaults = {
        "idx_name": "HNSW",
        "idx_type": "HNSW",
        "neighbors": 32,
        "efConstruction": 200,
        "accuracy": 90,
        "parallel": 8,
    }

    if params:
        config = params.copy()
        # Ensure compulsory parts are included
        for compulsory_key in ["idx_name", "parallel"]:
            if compulsory_key not in config:
                if compulsory_key == "idx_name":
                    config[compulsory_key] = _get_index_name(
                        str(defaults[compulsory_key])
                    )
                else:
                    config[compulsory_key] = defaults[compulsory_key]

        # Validate keys in config against defaults
        for key in config:
            if key not in defaults:
                raise ValueError(f"Invalid parameter: {key}")
    else:
        config = defaults

    # Base SQL statement
    idx_name = config["idx_name"]
    base_sql = (
        f"create vector index {idx_name} on {table_name}(embedding) "
        f"ORGANIZATION INMEMORY NEIGHBOR GRAPH"
    )

    # Optional parts depending on parameters
    accuracy_part = " WITH TARGET ACCURACY {accuracy}" if ("accuracy" in config) else ""
    distance_part = f" DISTANCE {distance_strategy}"

    parameters_part = ""
    if "neighbors" in config and "efConstruction" in config:
        parameters_part = (
            " parameters (type {idx_type}, neighbors {"
            "neighbors}, efConstruction {efConstruction})"
        )
    elif "neighbors" in config and "efConstruction" not in config:
        config["efConstruction"] = defaults["efConstruction"]
        parameters_part = (
            " parameters (type {idx_type}, neighbors {"
            "neighbors}, efConstruction {efConstruction})"
        )
    elif "neighbors" not in config and "efConstruction" in config:
        config["neighbors"] = defaults["neighbors"]
        parameters_part = (
            " parameters (type {idx_type}, neighbors {"
            "neighbors}, efConstruction {efConstruction})"
        )

    # Always included part for parallel
    parallel_part = " parallel {parallel}"

    # Combine all parts
    ddl_assembly = (
        base_sql + accuracy_part + distance_part + parameters_part + parallel_part
    )
    # Format the SQL with values from the params dictionary
    ddl = ddl_assembly.format(**config)

    # Check if the index exists
    if not await _index_exists(connection, config["idx_name"]):
        with connection.cursor() as cursor:
            await cursor.execute(ddl)
            logger.info("Index created successfully...")
    else:
        logger.info("Index already exists...")


@_handle_exceptions
async def _create_ivf_index(
    connection: Connection,
    table_name: str,
    distance_strategy: DistanceStrategy,
    params: Optional[dict[str, Any]] = None,
) -> None:
    # Default configuration
    defaults = {
        "idx_name": "IVF",
        "idx_type": "IVF",
        "neighbor_part": 32,
        "accuracy": 90,
        "parallel": 8,
    }

    if params:
        config = params.copy()
        # Ensure compulsory parts are included
        for compulsory_key in ["idx_name", "parallel"]:
            if compulsory_key not in config:
                if compulsory_key == "idx_name":
                    config[compulsory_key] = _get_index_name(
                        str(defaults[compulsory_key])
                    )
                else:
                    config[compulsory_key] = defaults[compulsory_key]

        # Validate keys in config against defaults
        for key in config:
            if key not in defaults:
                raise ValueError(f"Invalid parameter: {key}")
    else:
        config = defaults

    # Base SQL statement
    idx_name = config["idx_name"]
    base_sql = (
        f"CREATE VECTOR INDEX {idx_name} ON {table_name}(embedding) "
        f"ORGANIZATION NEIGHBOR PARTITIONS"
    )

    # Optional parts depending on parameters
    accuracy_part = " WITH TARGET ACCURACY {accuracy}" if ("accuracy" in config) else ""
    distance_part = f" DISTANCE {distance_strategy}"

    parameters_part = ""
    if "idx_type" in config and "neighbor_part" in config:
        parameters_part = (
            f" PARAMETERS (type {config['idx_type']}, neighbor"
            f" partitions {config['neighbor_part']})"
        )

    # Always included part for parallel
    parallel_part = f" PARALLEL {config['parallel']}"

    # Combine all parts
    ddl_assembly = (
        base_sql + accuracy_part + distance_part + parameters_part + parallel_part
    )
    # Format the SQL with values from the params dictionary
    ddl = ddl_assembly.format(**config)

    # Check if the index exists
    if not await _index_exists(connection, config["idx_name"]):
        with connection.cursor() as cursor:
            await cursor.execute(ddl)
        logger.info("Index created successfully...")
    else:
        logger.info("Index already exists...")


@_handle_exceptions
async def drop_table_purge(connection: Connection, table_name: str) -> None:
    """Drop a table and purge it from the database.

    Args:
        client: The OracleDB connection object.
        table_name: The name of the table to drop.

    Raises:
        RuntimeError: If an error occurs while dropping the table.
    """
    if await _table_exists(connection, table_name):
        with connection.cursor() as cursor:
            ddl = f"DROP TABLE {table_name} PURGE"
            await cursor.execute(ddl)
        logger.info("Table dropped successfully...")
    else:
        logger.info("Table not found...")
    return


class OracleVSMemoryConfig(BaseModel):
    """Base configuration for OracleVS-based memory implementation."""
    client: Any 
    params: Dict[str, str| int] #https://docs.oracle.com/en/database/oracle/oracle-database/23/arpls/dbms_vector_chain1.html#GUID-C6439E94-4E86-4ECD-954E-4B73D53579DE
    table_name: str
    modality: Literal["TEXT"] = Field(default="TEXT", description="Modality type")
    distance_strategy: Literal["dot", "euclidean", "cosine"] = Field(default="cosine", description="Distance metric for similarity search")
    k: int = Field(default=3, description="Number of results to return in queries")
    proxy: Optional[str | None] = Field(default=None, description="Query example????")


class OracleVSMemory(Memory, Component[OracleVSMemoryConfig]):

    component_type = "memory"
    component_config_schema = OracleVSMemoryConfig

    def __init__(self, config: OracleVSMemoryConfig) -> None:
        """Initialize OracleVSMemory."""
        self._config = config 
        self._embedding_dimension = None
        
        """Initialize with oracledb client."""
        self._client = self._config.client
        """Initialize with necessary components."""
        self._table_name = self._config.table_name
        self._distance_strategy = self._config.distance_strategy
        self._params = self._config.params
        self._proxy = self._config.proxy
        self._modality = self._config.modality
        self._k = self._config.k
        self._connection = None

    async def _ensure_initialized(self):

        if self._connection:
            return

        self._connection = await _get_connection(self._config.client)
        if self._connection is None:
            raise ValueError("Failed to acquire a connection.")
        _validate_version(self._connection)

        try:
            if self._proxy:
                with self._connection.cursor() as cursor:
                    await cursor.execute(
                        "begin utl_http.set_proxy(:proxy); end;", proxy=self._proxy
                    )

            await _create_table(self._connection, self._table_name, self._modality, await self._get_embedding_dimension())

        except oracledb.DatabaseError as db_err:
            logger.exception(f"Database error occurred while create table: {db_err}")
            raise RuntimeError(
                "Failed to create table due to a database error."
            ) from db_err
        except ValueError as val_err:
            logger.exception(f"Validation error: {val_err}")
            raise RuntimeError(
                "Failed to create table due to a validation error."
            ) from val_err
        except Exception as ex:
            logger.exception("An unexpected error occurred while creating the index.")
            raise RuntimeError(
                "Failed to create table due to an unexpected error."
            ) from ex

    async def _get_embedding_dimension(self):

        if self._embedding_dimension != None:
            return self._embedding_dimension

        ex_query = "Hello"
        ex_img_data = b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGK6brUBEAAA//8DtQHEz4SpmwAAAABJRU5ErkJggg=='

        model_input = ex_query

        if self._modality == "IMAGE": 
            model_input = ex_img_data

        with self._connection.cursor() as cursor:
            cursor.setinputsizes(None, oracledb.DB_TYPE_JSON)

            await cursor.execute(
                f"SELECT dbms_vector_chain.utl_to_embedding(:1, {"'image'," if self._modality=="IMAGE" else ""} json(:2))",
                (model_input, self._params),
            )

            res = await cursor.fetchone()

        self._embedding_dimension = len(res[0])
        return self._embedding_dimension

    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:

        # Get messages from context
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        # Extract query from last message
        last_message = messages[-1]
        query_text = last_message.content if isinstance(last_message.content, str) else str(last_message)

        # Query memory and get results
        query_results = await self.query(query_text)

        if query_results.results:
            # Format results for context
            memory_strings = [f"{i}. {str(memory.content)}" for i, memory in enumerate(query_results.results, 1)]
            memory_context = "\nRelevant memory content:\n" + "\n".join(memory_strings)

            # Add to context
            await model_context.add_message(SystemMessage(content=memory_context))

        return UpdateContextResult(memories=query_results)


    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:

        await self._ensure_initialized()
        if content.mime_type == MemoryMimeType.IMAGE and self._modality == "IMAGE":
            buffered = BytesIO()
            img = content.content.image
            img.save(buffered, format="JPEG")
            model_input = buffered.getvalue()

        elif content.mime_type == MemoryMimeType.TEXT and self._modality == "TEXT": 
            model_input = content.content
        else:
            raise ValueError("Only text and image supported, also must match")

        db_id = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:16].upper()

        with self._connection.cursor() as cursor:
            cursor.setinputsizes(None, None, oracledb.DB_TYPE_JSON, oracledb.DB_TYPE_JSON, None)
            await cursor.execute(
                f"INSERT INTO {self._table_name} (id, embedding, metadata, model_input) VALUES (:1, dbms_vector_chain.utl_to_embedding(:2,{"'image'," if self._modality=="IMAGE" else ""} json(:3)), :4, :5)",
                [db_id, model_input, self._params, {**content.metadata, "mime_type": content.mime_type.value},  model_input],
            )
            await self._connection.commit()

    def _extract_text(self, content_item: str | MemoryContent) -> str:
        """Extract searchable text from content."""
        if isinstance(content_item, str):
            return content_item

        content = content_item.content
        mime_type = content_item.mime_type

        # TODO: Should we support markdown
        if mime_type in [MemoryMimeType.TEXT]:#, MemoryMimeType.MARKDOWN]:
            return str(content)
        elif mime_type == MemoryMimeType.JSON:
            if isinstance(content, dict):
                # Store original JSON string representation
                return str(content).lower()
            raise ValueError("JSON content must be a dict")
        elif isinstance(content, Image):
            raise ValueError("Image content cannot be converted to text")
        else:
            raise ValueError(f"Unsupported content type: {mime_type}")

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        filter: Optional[FilterGroup] = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        #TODO Query should match model modality, but also doesnt have to match db vector modality - multimodal models

        await self._ensure_initialized()

        if cancellation_token is not None:
            logger.warning("cancellation_token not used")

        if isinstance(query,str) or (content.mime_type == MemoryMimeType.TEXT and self._modality == "TEXT"): 
            model_input = self._extract_text(query)

        elif content.mime_type == MemoryMimeType.IMAGE and self._modality == "IMAGE":
            # add with utl_embedding
            buffered = BytesIO()
            img = query.content.image
            img.save(buffered, format="JPEG")
            model_input = buffered.getvalue()

        else:
            raise ValueError("NOT CORRECT TYPE")

        if filter is not None:
            where_clause = _generate_where_clause(filter)

        with self._connection.cursor() as cursor:
            cursor.setinputsizes(None, oracledb.DB_TYPE_JSON)
            await cursor.execute(f"""
            SELECT id,
                model_input,
                metadata,
                vector_distance(embedding, dbms_vector_chain.utl_to_embedding(:1, {"'image'," if self._modality=="IMAGE" else ""} json(:2)), {self._distance_strategy}) as distance
            FROM {self._table_name}
            {f"WHERE {where_clause}" if filter is not None else ""}
            ORDER BY distance
            FETCH APPROX FIRST {self._k} ROWS ONLY
            """, [model_input, self._params])

            results = await cursor.fetchall()

            memory_results: List[MemoryContent] = []

            if not results:
                return MemoryQueryResult(results=memory_results)

            for result in results:
                metadata = dict(result[2])
                metadata["distance"] = result[3]

                mime_type = MemoryMimeType(str(metadata.get("mime_type", MemoryMimeType.TEXT.value)))

                del metadata["mime_type"]

                if mime_type == MemoryMimeType.TEXT:
                    doc = await _get_clob_value(result[1])
                elif mime_type == MemoryMimeType.IMAGE:
                    raise NotImplementedError

                # Create MemoryContent
                content = MemoryContent(
                    content=doc,
                    mime_type=mime_type,
                    metadata=metadata,
                )
                memory_results.append(content)

        return MemoryQueryResult(results=memory_results)


    async def create_index(self, params: Optional[dict[str, Any]] = None) -> None:
        await self._ensure_initialized()

        if self._connection is None:
            raise ValueError("Failed to acquire a connection.")
        if params:
            if params["idx_type"] == "HNSW":
                await _create_hnsw_index(
                    self._connection, 
                    self._table_name, 
                    self._distance_strategy, 
                    params
                )
            elif params["idx_type"] == "IVF":
                await _create_ivf_index(
                    self._connection, 
                    self._table_name, 
                    self._distance_strategy, 
                    params
                )
            else:
                raise ValueError("Only supported indexes HNSW and IVF")
        else:
            await _create_hnsw_index(
                self._connection, 
                self._table_name, 
                self._distance_strategy, 
                params
            )


    async def clear(self) -> None:
        await self._ensure_initialized()
        with self._connection.cursor() as cursor:
            await cursor.execute(
                f"TRUNCATE TABLE {self._table_name}",
            )
            await self._connection.commit()

    async def reset(self) -> None:
        await self._ensure_initialized()
        await drop_table_purge(self._connection, self._table_name)


    async def close(self) -> None:
        await self._ensure_initialized()

        if isinstance(self._connection, getattr(oracledb, "AsyncConnectionPool", None)):
            await self._connection.release()
        else:
            await self._connection.close()

    def _to_config(self) -> OracleVSMemoryConfig:
        """Serialize the memory configuration."""

        raise NotImplementedError

    @classmethod
    def _from_config(cls, config: OracleVSMemoryConfig) -> Self:
        """Deserialize the memory configuration."""

        return NotImplementedError



