from __future__ import annotations
import logging
import uuid
import functools
import hashlib
import json
import os
import io

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
    Literal
)

from autogen_core import CancellationToken, Component, Image
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
import numpy as np
from pydantic import BaseModel, Field
from autogen_core import Image


try:
    import oracledb
except ImportError as e:
    raise ImportError(
        "Unable to import oracledb, please install with "
        "`pip install -U oracledb`."
    ) from e


logger = logging.getLogger(__name__)
log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

T = TypeVar("T", bound=Callable[..., Any])

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


# =============================================================================
# DATABASE UTILS
# =============================================================================


async def _get_connection(client: Any) -> oracledb.Connection | oracledb.AsyncConnection | None:
    # check if AsyncConnectionPool exists
    connection_pool_class_async = getattr(oracledb, "AsyncConnectionPool", None)
    connection_pool_class_sync = getattr(oracledb, "ConnectionPool", None)

    if isinstance(client, oracledb.AsyncConnection) or isinstance(client, oracledb.Connection):
        return client
    elif connection_pool_class_async and isinstance(client, connection_pool_class_async):
        return await client.acquire()
    elif connection_pool_class_sync and isinstance(client, connection_pool_class_sync):
        return client.acquire()
    
    valid_types = "oracledb.AsyncConnection"
    valid_types += " or oracledb.Connection"
    if connection_pool_class_async:
        valid_types += " or oracledb.AsyncConnectionPool"
    if connection_pool_class_sync:
        valid_types += " or oracledb.ConnectionPool"
    raise TypeError(
        f"Expected client of type {valid_types}, got {type(client).__name__}"
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


async def _table_exists(connection: Union[oracledb.AsyncConnection, oracledb.Connection], table_name: str) -> bool:
    try:
        query = f"SELECT COUNT(*) FROM {table_name}"
        if isinstance(connection, oracledb.AsyncConnection):
            _ = await connection.fetchone(query)
        else:
            with connection.cursor() as cursor:
                cursor.execute(query)
                _ = cursor.fetchone()
        return True
    except oracledb.DatabaseError as ex:
        err_obj = ex.args
        if err_obj[0].code == 942:
            return False
        raise


async def _create_table(connection: Union[oracledb.AsyncConnection, oracledb.Connection], table_name: str, modality: str, embedding_dim: int) -> None:

    cols_dict = {
        "id": "RAW(16) DEFAULT SYS_GUID() PRIMARY KEY",
        "model_input": "CLOB" if modality == "TEXT" else "BLOB",
        "metadata": "JSON",
        "embedding": f"vector({embedding_dim})", #"vector"#
    }

    table_exists = await _table_exists(connection, table_name)

    if not table_exists:
        ddl_body = ", ".join(
                f"{col_name} {col_type}" for col_name, col_type in cols_dict.items()
            )
        ddl = f"CREATE TABLE {table_name} ({ddl_body})"
        if isinstance(connection, oracledb.AsyncConnection):
            await connection.execute(ddl)
        else:
            with connection.cursor() as cursor:
                cursor.execute(ddl)

        logger.info("Table created successfully...")
    else:
        logger.info("Table already exists...")


# =============================================================================
# FILTERING UTILS
# =============================================================================


class FilterCondition(TypedDict):
    key: str
    oper: str
    value: str


class FilterGroup(TypedDict, total=False):
    _and: Optional[List[Union[FilterCondition, FilterGroup]]]
    _or: Optional[List[Union[FilterCondition, FilterGroup]]]


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

# =============================================================================
# INDEXING UTILS
# =============================================================================

async def _index_exists(connection: Union[oracledb.AsyncConnection, oracledb.Connection], index_name: str) -> bool:
    # Check if the index exists
    query = """
        SELECT index_name 
        FROM all_indexes 
        WHERE upper(index_name) = upper(:idx_name)
        """
    with connection.cursor() as cursor:
        # Execute the query
        if isinstance(connection, oracledb.AsyncConnection):
            await cursor.execute(query, idx_name=index_name.upper())
            result = await cursor.fetchone()
        else:
            cursor.execute(query, idx_name=index_name.upper())
            result = cursor.fetchone()

    return result is not None

def _get_index_name(base_name: str) -> str:
    unique_id = str(uuid.uuid4()).replace("-", "")
    return f"{base_name}_{unique_id}"

async def _create_hnsw_index(
    connection: Union[oracledb.AsyncConnection, oracledb.Connection],
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
        if isinstance(connection, oracledb.AsyncConnection):
            await connection.execute(ddl)
        else:
            with connection.cursor() as cursor:
                cursor.execute(ddl)
        logger.info("Index created successfully...")
    else:
        logger.info("Index already exists...")


async def _create_ivf_index(
    connection: Union[oracledb.AsyncConnection, oracledb.Connection],
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
        if isinstance(connection, oracledb.AsyncConnection):
            await connection.execute(ddl)
        else:
            with connection.cursor() as cursor:
                cursor.execute(ddl)
        
        logger.info("Index created successfully...")
    else:
        logger.info("Index already exists...")

async def _image_to_lob(connection: Union[oracledb.AsyncConnection, oracledb.Connection], img: Any) -> oracledb.LOB | oracledb.AsyncLOB:

    try:
        from PIL import Image as PILImage
    except ImportError as e:
        raise ImportError(
            "Unable to import Pillow. It is required for image operations. Install with `pip install pillow`."
        ) from e

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    byte_image = buffered.getvalue()
    lob = (await connection.createlob(oracledb.DB_TYPE_BLOB, byte_image)) if isinstance(connection, oracledb.AsyncConnection) else connection.createlob(oracledb.DB_TYPE_BLOB, byte_image)

    return lob

# =============================================================================
# AUTOGEN UTILS
# =============================================================================

def _extract_text(content_item: str | MemoryContent) -> str:
    """Extract searchable text from content."""
    if isinstance(content_item, str):
        return content_item

    content = content_item.content
    mime_type = content_item.mime_type

    if mime_type in [MemoryMimeType.TEXT, MemoryMimeType.MARKDOWN]:
        return str(content)
    elif mime_type == MemoryMimeType.JSON:
        return json.dumps(content)
    else:
        raise ValueError(f"Unsupported content type: {mime_type}")

# =============================================================================
# AUTOGEN CLASSES
# =============================================================================

class OracleVSMemoryConfig(BaseModel):
    """Base configuration for OracleVS-based memory implementation."""
    client: Any 
    params: Dict[str, str| int] #https://docs.oracle.com/en/database/oracle/oracle-database/23/arpls/dbms_vector_chain1.html#GUID-C6439E94-4E86-4ECD-954E-4B73D53579DE
    table_name: str
    modality: Literal["TEXT", "IMAGE"] = Field(default="TEXT", description="Modality of the model, IMAGE or TEXT")
    distance_strategy: Literal["dot", "euclidean", "cosine"] = Field(default="cosine", description="Distance metric for similarity search")
    k: int = Field(default=3, description="Number of results to return in queries")
    proxy: Optional[str | None] = Field(default=None, description="Proxy for external providers")


class OracleVSMemory(Memory, Component[OracleVSMemoryConfig]):

    component_type = "memory"
    component_config_schema = OracleVSMemoryConfig

    def __init__(self, config: OracleVSMemoryConfig) -> None:
        """Initialize OracleVSMemory."""
        self._config = config 
        self._embedding_dimension = None
        self._client = self._config.client
        self._table_name = self._config.table_name
        self._distance_strategy = self._config.distance_strategy
        self._params = self._config.params
        self._proxy = self._config.proxy
        self._modality = self._config.modality
        self._k = self._config.k
        self._connection = None
        self._is_async = None

    @_handle_exceptions
    async def _ensure_initialized(self):

        if self._connection: return

        self._connection = await _get_connection(self._config.client)
        if self._connection is None:
            raise ValueError("Failed to acquire a connection.")

        if _compare_version(oracledb.__version__, "2.2.0"):
            raise Exception(
                f"Oracle DB python client driver version {oracledb.__version__} not supported, must be >=2.2.0 for vector support"
            )

        if not(hasattr(self._connection, "thin") and self._connection.thin):
            if oracledb.clientversion()[:2] < (23,4):
                raise Exception(
                    f"Oracle DB client driver version {oracledb.__version__} not supported, must be >=23.4 for vector support"
                )

        self._is_async = False
        if isinstance(self._connection, oracledb.AsyncConnection):
            self._is_async = True

        if self._proxy:
            query = "begin utl_http.set_proxy(:proxy); end;"
            with self._connection.cursor() as cursor:
                (await cursor.execute(
                    query, proxy=self._proxy
                )) if self._is_async else cursor.execute(
                    query, proxy=self._proxy
                )

        await _create_table(self._connection, self._table_name, self._modality, await self._get_embedding_dimension())

    @_handle_exceptions
    async def _get_clob_value(self, result: Any) -> str:
        clob_value = ""
        if result:
            if isinstance(result, oracledb.AsyncLOB):
                raw_data = (await result.read()) if self._is_async else result.read()
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
    async def _get_embedding_dimension(self):

        if self._modality == "IMAGE":
            try:
                from PIL import Image as PILImage
            except ImportError as e:
                raise ImportError(
                    "Unable to import Pillow. It is required for image operations. Install with `pip install pillow`."
                ) from e

        if self._embedding_dimension != None:
            return self._embedding_dimension

        ex_query = "Hello"
        model_input = ex_query

        if self._modality == "IMAGE": 
            img = PILImage.new("RGB", (10, 10))
            ex_img_data = await _image_to_lob(self._connection, img)
            model_input = ex_img_data

        with self._connection.cursor() as cursor:
            cursor.setinputsizes(None, oracledb.DB_TYPE_JSON)

            query = f"SELECT dbms_vector_chain.utl_to_embedding(:1, {"'image'," if self._modality=="IMAGE" else ""} json(:2))"
            bind_input = [model_input, self._params]

            (await cursor.execute(
                query,
                bind_input,
            )) if self._is_async else cursor.execute(
                query,
                bind_input,
            )

            res = (await cursor.fetchone()) if self._is_async else cursor.fetchone()

        self._embedding_dimension = len(res[0])
        return self._embedding_dimension

    @_handle_exceptions
    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:

        await self._ensure_initialized()

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

    @_handle_exceptions
    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:

        await self._ensure_initialized()

        if cancellation_token and cancellation_token.is_cancelled():
            logger.info("Message addition cancelled by token.")
            return 

        if content.mime_type == MemoryMimeType.IMAGE and self._modality == "IMAGE":
            model_input = await _image_to_lob(self._connection, content.content.image)
        elif self._modality == "TEXT": 
            model_input = _extract_text(content)
        else:
            raise ValueError("Only text and image supported, also must match")

        db_id = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:16].upper()

        with self._connection.cursor() as cursor:
            cursor.setinputsizes(None, None, oracledb.DB_TYPE_JSON, oracledb.DB_TYPE_JSON, None)

            query = f"INSERT INTO {self._table_name} (id, embedding, metadata, model_input) VALUES (:1, dbms_vector_chain.utl_to_embedding(:2,{"'image'," if self._modality=="IMAGE" else ""} json(:3)), :4, :5)"
            bind_input = [db_id, model_input, self._params, {**content.metadata, "mime_type": content.mime_type.value},  model_input]

            (await cursor.execute(
                query,
                bind_input,
            )) if self._is_async else cursor.execute(
                query,
                bind_input,
            )

            (await self._connection.commit()) if self._is_async else self._connection.commit()

    @_handle_exceptions
    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        filter: Optional[FilterGroup] = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:

        await self._ensure_initialized()

        if cancellation_token and cancellation_token.is_cancelled():
            logger.info("Message query cancelled by token.")
            return MemoryQueryResult(results=[]) 

        if self._modality == "IMAGE":
            if not (isinstance(query,MemoryContent) and query.mime_type == MemoryMimeType.IMAGE):
                raise ValueError("Image models can only be queried with MemoryContent")
            model_input = await _image_to_lob(self._connection, query.content.image)
        elif self._modality == "TEXT": 
            model_input = _extract_text(query)

        if filter is not None:
            where_clause = _generate_where_clause(filter)

        with self._connection.cursor() as cursor:
            cursor.setinputsizes(None, oracledb.DB_TYPE_JSON)

            query = f"""
            SELECT id,
                model_input,
                metadata,
                vector_distance(embedding, dbms_vector_chain.utl_to_embedding(:1, {"'image'," if self._modality=="IMAGE" else ""} json(:2)), {self._distance_strategy}) as distance
            FROM {self._table_name}
            {f"WHERE {where_clause}" if filter is not None else ""}
            ORDER BY distance
            FETCH APPROX FIRST {self._k} ROWS ONLY
            """

            bind_input = [model_input, self._params]

            (await cursor.execute(query, bind_input)) if self._is_async else cursor.execute(query, bind_input)

            results = (await cursor.fetchall()) if self._is_async else cursor.fetchall()

            memory_results: List[MemoryContent] = []

            if not results:
                return MemoryQueryResult(results=memory_results)

            for result in results:
                metadata = dict(result[2])
                metadata["distance"] = result[3]

                mime_type = MemoryMimeType(str(metadata.get("mime_type", MemoryMimeType.TEXT.value)))

                del metadata["mime_type"]

                if mime_type in [MemoryMimeType.TEXT, MemoryMimeType.MARKDOWN]:
                    doc = await self._get_clob_value(result[1])
                if mime_type == MemoryMimeType.JSON:
                    doc = json.loads(await self._get_clob_value(result[1]))
                elif mime_type == MemoryMimeType.IMAGE:
                    try:
                        from PIL import Image as PILImage
                    except ImportError as e:
                        raise ImportError(
                            "Unable to import Pillow. It is required for image operations. Install with `pip install pillow`."
                        ) from e

                    # TODO: Should we expect images smaller than 1GB always - https://python-oracledb.readthedocs.io/en/latest/user_guide/lob_data.html#streaming-lobs-read
                    image_bytes = b""
                    offset = 1
                    num_bytes_in_chunk = 65536  
                    while True:
                        data = (await result[1].read(offset, num_bytes_in_chunk)) if self._is_async else result[1].read(offset, num_bytes_in_chunk)
                        if data:
                            image_bytes += data 
                        if len(data) < num_bytes_in_chunk:
                            break
                        offset += len(data)

                    image_stream = io.BytesIO(image_bytes)

                    doc = Image.from_pil(PILImage.open(image_stream))

                # Create MemoryContent
                content = MemoryContent(
                    content=doc,
                    mime_type=mime_type,
                    metadata=metadata,
                )
                memory_results.append(content)

        return MemoryQueryResult(results=memory_results)

    @_handle_exceptions
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

    @_handle_exceptions
    async def clear(self) -> None:
        await self._ensure_initialized()
        query = f"TRUNCATE TABLE {self._table_name}"
        if self._is_async:
            await self._connection.execute(query)
        else:
            with self._connection.cursor() as cursor:
                cursor.execute(query)


    @_handle_exceptions
    async def close(self) -> None:
        # Responsibility of the user 
        await self._ensure_initialized()

        if isinstance(self._config.client, getattr(oracledb, "AsyncConnectionPool", None)):
            (await self._config.client.release(self._connection)) if self._is_async else self._config.client.release(self._connection)

    def _to_config(self) -> OracleVSMemoryConfig:
        """Serialize the memory configuration."""

        raise NotImplementedError

    @classmethod
    def _from_config(cls, config: OracleVSMemoryConfig) -> Self:
        """Deserialize the memory configuration."""

        return NotImplementedError



