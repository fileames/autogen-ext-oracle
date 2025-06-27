from pathlib import Path

import pytest
import pytest_asyncio
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import UserMessage
from autogen_ext_oracle import OracleVSMemory, OracleVSMemoryConfig
from autogen_core import CancellationToken
from typing import AsyncGenerator
from typing import (
    AsyncGenerator,
    Tuple,
    Dict,
    List
)
from PIL import Image as PILImage
from autogen_core import Image

# Skip all tests if oracledb is not available
try:
    import oracledb  # pyright: ignore[reportUnusedImport]
except ImportError:
    pytest.skip("oracledb not available", allow_module_level=True)
    

username = "user"
password = "user"
dsn = "dsn"

async def drop_table_purge(connection, table_name) -> None:
    ddl = f"DROP TABLE IF EXISTS {table_name} PURGE"

    if isinstance(connection, oracledb.AsyncConnection):
        await connection.execute(ddl)
    else:
        with connection.cursor() as cursor:
            cursor.execute(ddl)


@pytest_asyncio.fixture()
async def txt_config() -> AsyncGenerator[OracleVSMemoryConfig]:
    """Base configuration."""

    connection = await oracledb.connect_async(user=username, password=password, dsn=dsn)

    config=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider" : "database", 
            "model"    : "allminilm" 
            },
        table_name="mytable",
        modality="TEXT",
        distance_strategy="cosine",
    )

    yield config

    await drop_table_purge(connection, "mytable")
    await connection.close()

@pytest_asyncio.fixture()
async def txt_2conn_config() -> AsyncGenerator[Tuple[OracleVSMemoryConfig, OracleVSMemoryConfig]]:
    """Two configs with same table, two connections."""

    connection = await oracledb.connect_async(user=username, password=password, dsn=dsn)

    config=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider" : "database", 
            "model"    : "allminilm" 
            },
        table_name="mytable",
        modality="TEXT",
        distance_strategy="cosine",
    )

    connection2 = await oracledb.connect_async(user=username, password=password, dsn=dsn)

    config2=OracleVSMemoryConfig(
        client=connection2,
        params = {
            "provider" : "database", 
            "model"    : "allminilm" 
            },
        table_name="mytable",
        modality="TEXT",
        distance_strategy="cosine",
    )

    yield (config, config2)

    await drop_table_purge(connection, "mytable")
    await connection.close()
    await connection2.close()

@pytest_asyncio.fixture()
async def txt_sync_2conn_config() -> AsyncGenerator[Tuple[OracleVSMemoryConfig, OracleVSMemoryConfig]]:
    """Two configs with same table, two connections."""

    connection = oracledb.connect(user=username, password=password, dsn=dsn)

    config=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider" : "database", 
            "model"    : "allminilm" 
            },
        table_name="mytable",
        modality="TEXT",
        distance_strategy="cosine",
    )

    connection2 = oracledb.connect(user=username, password=password, dsn=dsn)

    config2=OracleVSMemoryConfig(
        client=connection2,
        params = {
            "provider" : "database", 
            "model"    : "allminilm" 
            },
        table_name="mytable",
        modality="TEXT",
        distance_strategy="cosine",
    )

    yield (config, config2)

    await drop_table_purge(connection, "mytable")
    connection.close()
    connection2.close()



@pytest_asyncio.fixture()
async def txt_conn_config() -> AsyncGenerator[OracleVSMemoryConfig]:
    """Setup without closing connections, to test closed, broken connections"""

    connection = await oracledb.connect_async(user=username, password=password, dsn=dsn)

    config=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider" : "database", 
            "model"    : "allminilm" 
            },
        table_name="mytable",
        modality="TEXT",
        distance_strategy="cosine",
    )

    yield config

@pytest_asyncio.fixture()
async def txt_sync_conn_config() -> AsyncGenerator[OracleVSMemoryConfig]:
    """Setup without closing connections, to test closed, broken connections"""

    connection = oracledb.connect(user=username, password=password, dsn=dsn)

    config=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider" : "database", 
            "model"    : "allminilm" 
            },
        table_name="mytable",
        modality="TEXT",
        distance_strategy="cosine",
    )

    yield config


@pytest_asyncio.fixture()
async def txt_sync_config() -> AsyncGenerator[OracleVSMemoryConfig]:
    """Synchronous connection - should not work"""

    connection = oracledb.connect(user=username, password=password, dsn=dsn)

    config=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider" : "database", 
            "model"    : "allminilm" 
            },
        table_name="mytable",
        modality="TEXT",
        distance_strategy="cosine",
    )

    yield config

    await drop_table_purge(connection, "mytable")
    connection.close()

@pytest_asyncio.fixture()
async def txt_pool_config() -> AsyncGenerator[Tuple[OracleVSMemoryConfig, OracleVSMemoryConfig]]:
    """Two configs using the same async pool"""

    connection = oracledb.create_pool_async(user=username, password=password, dsn=dsn, min=1, max=4, increment=1)

    config=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider" : "database", 
            "model"    : "allminilm" 
            },
        table_name="mytable",
        modality="TEXT",
        distance_strategy="cosine",
    )

    config2=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider" : "database", 
            "model"    : "allminilm" 
            },
        table_name="mytable2",
        modality="TEXT",
        distance_strategy="cosine",
    )

    yield (config, config2)

    async with connection.acquire() as _conn:
        await drop_table_purge(_conn, "mytable")
        await drop_table_purge(_conn, "mytable2")

    await connection.close()

@pytest_asyncio.fixture()
async def txt_sync_pool_config() -> AsyncGenerator[Tuple[OracleVSMemoryConfig, OracleVSMemoryConfig]]:
    """Two configs using the same async pool"""

    connection = oracledb.create_pool(user=username, password=password, dsn=dsn, min=1, max=4, increment=1)

    config=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider" : "database", 
            "model"    : "allminilm" 
            },
        table_name="mytable",
        modality="TEXT",
        distance_strategy="cosine",
    )

    config2=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider" : "database", 
            "model"    : "allminilm" 
            },
        table_name="mytable2",
        modality="TEXT",
        distance_strategy="cosine",
    )

    yield (config, config2)

    with connection.acquire() as _conn:
        await drop_table_purge(_conn, "mytable")
        await drop_table_purge(_conn, "mytable2")

    connection.close()

@pytest_asyncio.fixture()
async def txt_oci_config() -> AsyncGenerator[OracleVSMemoryConfig]:
    """Base configuration."""

    connection = await oracledb.connect_async(user=username, password=password, dsn=dsn)

    config=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider"       : "ocigenai",
            "credential_name": "OCI_CRED",
            "url"            : "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
            "model"          : "cohere.embed-english-v3.0",
            "batch_size"     : 10
        },
        table_name="mytable",
        modality="TEXT",
        distance_strategy="cosine",
    )

    yield config

    await drop_table_purge(connection, "mytable")
    await connection.close()

@pytest_asyncio.fixture()
async def txt_oci_sync_config() -> AsyncGenerator[OracleVSMemoryConfig]:
    """Base configuration."""

    connection = oracledb.connect(user=username, password=password, dsn=dsn)

    config=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider"       : "ocigenai",
            "credential_name": "OCI_CRED",
            "url"            : "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
            "model"          : "cohere.embed-english-v3.0",
            "batch_size"     : 10
        },
        table_name="mytable",
        modality="TEXT",
        distance_strategy="cosine",
    )

    yield config

    await drop_table_purge(connection, "mytable")
    connection.close()

@pytest_asyncio.fixture()
async def txt_cohere_config() -> AsyncGenerator[OracleVSMemoryConfig]:
    """Base configuration."""

    connection = await oracledb.connect_async(user=username, password=password, dsn=dsn)

    config=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider"       : "cohere",
            "credential_name": "COHERE_CRED",
            "url"            : "https://api.cohere.ai/v1/embed",
            "model"          : "embed-english-light-v3.0",
            "input_type"     : "search_query"
        },
        table_name="mytable",
        modality="TEXT",
        distance_strategy="cosine",
        proxy="www-proxy-ash7.us.oracle.com:80"
    )


    yield config

    await drop_table_purge(connection, "mytable")
    await connection.close()

@pytest_asyncio.fixture()
async def multimodal_cohere_config() -> AsyncGenerator[Tuple[OracleVSMemoryConfig, OracleVSMemoryConfig]]:
    """Base configuration."""

    connection = await oracledb.connect_async(user=username, password=password, dsn=dsn)

    config=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider"       : "cohere",
            "credential_name": "COHERE_CRED",
            "url"            : "https://api.cohere.ai/v2/embed",
            "model"          : "embed-v4.0",
        },
        table_name="mytable",
        modality="IMAGE",
        distance_strategy="cosine",
        proxy="www-proxy-ash7.us.oracle.com:80",
        k=1
    )

    config2=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider"       : "cohere",
            "credential_name": "COHERE_CRED",
            "url"            : "https://api.cohere.ai/v2/embed",
            "model"          : "embed-v4.0",
        },
        table_name="mytable",
        modality="TEXT",
        distance_strategy="cosine",
        proxy="www-proxy-ash7.us.oracle.com:80",
        k=1
    )


    yield config,config2

    await drop_table_purge(connection, "mytable")
    await connection.close()


@pytest_asyncio.fixture()
async def image_cohere_config() -> AsyncGenerator[OracleVSMemoryConfig]:
    """Base configuration."""

    connection = await oracledb.connect_async(user=username, password=password, dsn=dsn)

    config=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider"       : "cohere",
            "credential_name": "COHERE_CRED",
            "url"            : "https://api.cohere.ai/v2/embed",
            "model"          : "embed-v4.0",
        },
        table_name="mytable",
        modality="IMAGE",
        distance_strategy="cosine",
        proxy="www-proxy-ash7.us.oracle.com:80",
        k=1
    )

    yield config

    await drop_table_purge(connection, "mytable")
    await connection.close()

@pytest_asyncio.fixture()
async def image_cohere_sync_config() -> AsyncGenerator[OracleVSMemoryConfig]:
    """Base configuration."""

    connection = oracledb.connect(user=username, password=password, dsn=dsn)

    config=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider"       : "cohere",
            "credential_name": "COHERE_CRED",
            "url"            : "https://api.cohere.ai/v2/embed",
            "model"          : "embed-v4.0",
        },
        table_name="mytable",
        modality="IMAGE",
        distance_strategy="cosine",
        proxy="www-proxy-ash7.us.oracle.com:80",
        k=1
    )

    yield config

    await drop_table_purge(connection, "mytable")
    connection.close()

@pytest.fixture
def img_data() -> Tuple[List, Dict]:
    return ([
        {"fruit_type": "apple", "url": "resources/42126750_9579e4e830_z.jpg"},
        {"fruit_type": "strawberry", "url": "resources/18646827786_9f316d4cd1_z.jpg"},
        {"fruit_type": "watermelon", "url": "resources/4688473695_452e494f5c_z.jpg"},
    ], {"fruit_type": "watermelon", "url": "resources/5055320652_e126cac865_z.jpg"})

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize('txt_config', ['txt_config', 'txt_sync_config'], indirect=True)
async def test_basic_workflow(txt_config: OracleVSMemoryConfig) -> None:
    """Test basic query"""
    memory = OracleVSMemory(txt_config)
    await memory.clear()

    await memory.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        )
    )

    results = await memory.query("Tell me about Paris")
    assert len(results.results) > 0
    assert any("Paris" in str(r.content) for r in results.results)
    assert all(isinstance(r.metadata.get("distance"), float) for r in results.results if r.metadata)

@pytest.mark.asyncio
@pytest.mark.parametrize('txt_config', ['txt_config', 'txt_sync_config'], indirect=True)
async def test_proxy(txt_config: OracleVSMemoryConfig) -> None:
    """Test that proxy setting does not produce errors"""
    txt_config.proxy = 'proxy.my-company.com'
    memory = OracleVSMemory(txt_config)
    await memory.clear()

    await memory.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        )
    )

    results = await memory.query("Tell me about Paris")
    assert len(results.results) > 0
    assert any("Paris" in str(r.content) for r in results.results)
    assert all(isinstance(r.metadata.get("distance"), float) for r in results.results if r.metadata)


@pytest.mark.asyncio
@pytest.mark.parametrize('txt_2conn_config', ['txt_2conn_config', 'txt_sync_2conn_config'], indirect=True)
async def test_same_connection_workflow(txt_2conn_config: Tuple[OracleVSMemoryConfig, OracleVSMemoryConfig]) -> None:
    """Same table name two connections. Test that OracleVSMemory can see the changes."""
    memory = OracleVSMemory(txt_2conn_config[0])
    await memory.clear()

    results = await memory.query("Tell me about Paris")
    assert len(results.results) == 0

    await memory.add(
        MemoryContent(
            content="Istanbul is known for its rich history, diverse culture.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "Turkiye"},
        )
    )

    memory2 = OracleVSMemory(txt_2conn_config[1])

    await memory2.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        )
    )

    results = await memory.query("Tell me about Paris")
    assert len(results.results) > 0
    assert any("Paris" in str(r.content) for r in results.results)
    assert all(isinstance(r.metadata.get("distance"), float) for r in results.results if r.metadata)

@pytest.mark.asyncio
@pytest.mark.parametrize('txt_config', ['txt_config', 'txt_sync_config'], indirect=True)
async def test_hnsw_index(txt_config: OracleVSMemoryConfig) -> None:
    """Test hnsw index creation"""
    memory = OracleVSMemory(txt_config)
    await memory.clear()

    await memory.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        )
    )

    await memory.add(
        MemoryContent(
            content="Istanbul is known for its rich history, diverse culture.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "Turkiye"},
        )
    )

    await memory.create_index(params={"idx_name": "hnsw_idx1", "idx_type": "HNSW"})

    results = await memory.query("Tell me about Istanbul")
    assert len(results.results) > 0
    assert any("Paris" in str(r.content) for r in results.results)
    assert all(isinstance(r.metadata.get("distance"), float) for r in results.results if r.metadata)


@pytest.mark.asyncio
@pytest.mark.parametrize('txt_config', ['txt_config', 'txt_sync_config'], indirect=True)

async def test_ivf_index(txt_config: OracleVSMemoryConfig) -> None:
    """Test ivf index creation"""
    memory = OracleVSMemory(txt_config)
    await memory.clear()

    await memory.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        )
    )

    await memory.add(
        MemoryContent(
            content="Istanbul is known for its rich history, diverse culture.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "Turkiye"},
        )
    )

    await memory.create_index(params={
            "idx_name": "ivf_idx2",
            "idx_type": "IVF",
            "accuracy": 90,
            "parallel": 32,
        })

    results = await memory.query("Tell me about Istanbul")
    assert len(results.results) > 0
    assert any("Paris" in str(r.content) for r in results.results)
    assert all(isinstance(r.metadata.get("distance"), float) for r in results.results if r.metadata)


@pytest.mark.asyncio
@pytest.mark.parametrize('txt_conn_config', ['txt_conn_config', 'txt_sync_conn_config'], indirect=True)
async def test_error_after_close(txt_conn_config: OracleVSMemoryConfig) -> None:
    """Test closed connection"""
    memory = OracleVSMemory(txt_conn_config)
    await memory.clear()

    await memory.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        )
    )

    (await txt_conn_config.client.close()) if isinstance(txt_conn_config.client, (oracledb.AsyncConnection, oracledb.AsyncConnectionPool)) else txt_conn_config.client.close()

    with pytest.raises(oracledb.Error):
        await memory.add(
            MemoryContent(
                content="Istanbul is known for its rich history, diverse culture.",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "city", "country": "Turkiye"},
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('txt_config', ['txt_config', 'txt_sync_config'], indirect=True)
async def test_clear(txt_config: OracleVSMemoryConfig) -> None:
    """Test clear - table should be cleared"""
    memory = OracleVSMemory(txt_config)
    await memory.clear()

    await memory.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        )
    )

    results = await memory.query("Tell me about Istanbul")
    assert len(results.results) > 0

    await memory.clear()

    results = await memory.query("Tell me about Istanbul")
    assert len(results.results) == 0

    
@pytest.mark.asyncio
@pytest.mark.parametrize('txt_config', ['txt_config', 'txt_sync_config'], indirect=True)
async def test_content_types_error(txt_config: OracleVSMemoryConfig) -> None:
    """Test content types other than TEXT and IMAGE, which are not allowed"""
    memory = OracleVSMemory(txt_config)
    await memory.clear()

    with pytest.raises(ValueError): #, match="Unsupported content type"):
        await memory.add(MemoryContent(content=b"binary data", mime_type=MemoryMimeType.BINARY))

    '''with pytest.raises(ValueError):
        await memory.add(MemoryContent(content="not a dict", mime_type=MemoryMimeType.JSON))'''

    # TODO: what to do with images

@pytest.mark.asyncio
@pytest.mark.parametrize('txt_config', ['txt_config', 'txt_sync_config'], indirect=True)
async def test_content_types(txt_config: OracleVSMemoryConfig) -> None:
    """Test content types"""
    memory = OracleVSMemory(txt_config)
    await memory.clear()

    await memory.add(
        MemoryContent(
            content="""
            # Paris: The City of Light
            Paris, the capital of France, is known for its iconic landmarks like the **Eiffel Tower**, **Louvre Museum**, and **Notre-Dame Cathedral**. 

            ## Must-See:
            - **Eiffel Tower**
            - **Louvre**
            - **Notre-Dame**

            Paris is a city of romance, beauty, and timeless charm.

            """,
            mime_type=MemoryMimeType.MARKDOWN,
            metadata={"category": "city", "country": "France"},
        )
    )

    results = await memory.query("Tell me about Paris")
    assert len(results.results) > 0
    assert results.results[0].mime_type == MemoryMimeType.MARKDOWN
    assert any("Paris" in str(r.content) for r in results.results)

    await memory.clear()

    await memory.add(
        MemoryContent(
            content={
            "city": "Paris",
            "nickname": "The City of Light",
            "country": "France",
            "highlights": [
                "Eiffel Tower",
                "Louvre Museum",
                "Notre-Dame Cathedral"
            ],
            "description": "Paris is known for its iconic landmarks, art, culture, and fashion. ",
            },
            mime_type=MemoryMimeType.JSON,
            metadata={"category": "city", "country": "France"},
        )
    )

    results = await memory.query("Tell me about Paris")
    assert len(results.results) > 0
    assert results.results[0].mime_type == MemoryMimeType.JSON
    assert isinstance(results.results[0].content, Dict) and results.results[0].content["city"] == "Paris"

@pytest.mark.asyncio
@pytest.mark.parametrize('txt_config', ['txt_config', 'txt_sync_config'], indirect=True)
async def test_content_types_mix(txt_config: OracleVSMemoryConfig) -> None:
    """Test content types"""
    memory = OracleVSMemory(txt_config)
    await memory.clear()

    await memory.add(
        MemoryContent(
            content={
            "city": "Paris",
            "nickname": "The City of Light",
            "country": "France",
            "highlights": [
                "Eiffel Tower",
                "Louvre Museum",
                "Notre-Dame Cathedral"
            ],
            "description": "Paris is known for its iconic landmarks, art, culture, and fashion. ",
            },
            mime_type=MemoryMimeType.JSON,
            metadata={"category": "city", "country": "France"},
        )
    )

    results = await memory.query(MemoryContent(
            content="""
            # Paris: The City of Light

            Must see places:
            - ...
            """,
            mime_type=MemoryMimeType.MARKDOWN,
            metadata={"category": "city", "country": "France"},
        )
    )

    assert len(results.results) > 0
    assert results.results[0].mime_type == MemoryMimeType.JSON
    assert isinstance(results.results[0].content, Dict) and results.results[0].content["city"] == "Paris"


@pytest.mark.asyncio
@pytest.mark.parametrize('txt_config', ['txt_config', 'txt_sync_config'], indirect=True)
async def test_metadata_handling(txt_config: OracleVSMemoryConfig) -> None:
    """Test metadata handling"""
    memory = OracleVSMemory(txt_config)
    await memory.clear()

    test_content = "Test content with specific metadata"
    content = MemoryContent(
        content=test_content,
        mime_type=MemoryMimeType.TEXT,
        metadata={"test_category": "test", "test_priority": 1, "test_weight": 0.5, "test_verified": True},
    )
    await memory.add(content)

    results = await memory.query(test_content)
    assert len(results.results) > 0
    result = results.results[0]

    assert result.metadata is not None
    assert result.metadata.get("test_category") == "test"
    assert result.metadata.get("test_priority") == 1
    assert result.metadata.get("test_weight") == pytest.approx(0.5)
    assert result.metadata.get("test_verified") is True


@pytest.mark.asyncio
@pytest.mark.parametrize('txt_pool_config', ['txt_pool_config', 'txt_sync_pool_config'], indirect=True)
async def test_pool(txt_pool_config: Tuple[OracleVSMemoryConfig,OracleVSMemoryConfig]) -> None:
    """Test pool connection"""
    memory = OracleVSMemory(txt_pool_config[0])
    await memory.clear()
    await memory.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        )
    )

    memory2 = OracleVSMemory(txt_pool_config[1])
    await memory2.clear()

    await memory2.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        )
    )

    assert txt_pool_config[0].client.busy == 2

    await memory.close()

    assert txt_pool_config[0].client.busy == 1

    with pytest.raises(oracledb.InterfaceError, match="not connected to database"):
        await memory.add(
            MemoryContent(
                content="Paris is known for the Eiffel Tower and amazing cuisine.",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "city", "country": "France"},
            )
        )

    await memory2.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        )
    )

    await memory2.close()
    
@pytest.mark.asyncio
@pytest.mark.parametrize('txt_config', ['txt_config', 'txt_sync_config'], indirect=True)
async def test_model_context_update(txt_config: OracleVSMemoryConfig) -> None:
    """Test updating model context with retrieved memories"""
    memory = OracleVSMemory(txt_config)
    await memory.clear()

    # Add content to memory
    await memory.add(
        MemoryContent(
            content="Jupiter is the largest planet in our solar system.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "astronomy"},
        )
    )

    # Create a model context with a message
    context = BufferedChatCompletionContext(buffer_size=5)
    await context.add_message(UserMessage(content="Tell me about Jupiter", source="user"))

    # Update context with memory
    result = await memory.update_context(context)

    # Verify results
    assert len(result.memories.results) > 0
    assert any("Jupiter" in str(r.content) for r in result.memories.results)

    # Verify context was updated
    messages = await context.get_messages()
    assert len(messages) > 1  # Should have the original message plus the memory content


@pytest.mark.asyncio
@pytest.mark.parametrize('txt_config', ['txt_config', 'txt_sync_config'], indirect=True)
async def test_filter(txt_config: OracleVSMemoryConfig) -> None:
    """Test database filtering"""
    memory = OracleVSMemory(txt_config)
    await memory.clear()

    await memory.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        )
    )

    await memory.add(
        MemoryContent(
            content="Istanbul is known for its rich history, diverse culture.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "Turkiye"},
        )
    )

    db_filter:  FilterGroup = {
        "_and": [
        {"key": "country", "oper": "EQ", "value": "Turkiye"},  
      ]
    }

    results = await memory.query("Tell me about Paris", filter=db_filter)
    assert len(results.results) > 0
    assert all("Paris" not in str(r.content) for r in results.results)
    assert all(isinstance(r.metadata.get("distance"), float) for r in results.results if r.metadata)

@pytest.mark.asyncio
@pytest.mark.parametrize('txt_config', ['txt_config', 'txt_sync_config'], indirect=True)
async def test_cancellation(txt_config: OracleVSMemoryConfig) -> None:
    """Test cancellation token operations."""
    memory = OracleVSMemory(txt_config)
    await memory.clear()

    token = CancellationToken()
    token.cancel()

    await memory.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        ),
        cancellation_token = token
    )

    results = await memory.query("Tell me about Paris")
    assert len(results.results) == 0

    await memory.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        )
    )

    results = await memory.query("Tell me about Paris")
    assert len(results.results) == 1

    results = await memory.query("Tell me about Paris", cancellation_token = token)
    assert len(results.results) == 0

@pytest.mark.asyncio
@pytest.mark.parametrize('txt_oci_config', ['txt_oci_config', 'txt_oci_sync_config'], indirect=True)
async def test_oci_workflow(txt_oci_config: OracleVSMemoryConfig) -> None:
    """Test basic query"""
    memory = OracleVSMemory(txt_oci_config)
    await memory.clear()

    results = await memory.query("Tell me about Paris")
    assert len(results.results) == 0

    await memory.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        )
    )

    results = await memory.query("Tell me about Paris")
    assert len(results.results) > 0
    assert any("Paris" in str(r.content) for r in results.results)
    assert all(isinstance(r.metadata.get("distance"), float) for r in results.results if r.metadata)

@pytest.mark.asyncio
async def test_cohere_workflow(txt_cohere_config: OracleVSMemoryConfig) -> None:
    """Test basic query"""
    memory = OracleVSMemory(txt_cohere_config)
    await memory.clear()

    results = await memory.query("Tell me about Paris")
    assert len(results.results) == 0

    await memory.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        )
    )

    results = await memory.query("Tell me about Paris")
    assert len(results.results) > 0
    assert any("Paris" in str(r.content) for r in results.results)
    assert all(isinstance(r.metadata.get("distance"), float) for r in results.results if r.metadata)

@pytest.mark.asyncio 
@pytest.mark.parametrize('image_cohere_config', ['image_cohere_config', 'image_cohere_sync_config'], indirect=True)
async def test_cohere_image_workflow(image_cohere_config: OracleVSMemoryConfig, img_data: Tuple[List, Dict]) -> None:
    """Test basic query"""
    memory = OracleVSMemory(image_cohere_config)
    await memory.clear()

    for doc in img_data[0]:
        await memory.add(
            MemoryContent(
                content= Image.from_pil(PILImage.open(doc["url"])),
                mime_type=MemoryMimeType.IMAGE,
                metadata={"fruit_type": doc["fruit_type"]},
            )
        )

    results = await memory.query(MemoryContent(
                content= Image.from_pil(PILImage.open(img_data[1]["url"])),
                mime_type=MemoryMimeType.IMAGE
            )
    )

    assert len(results.results) == 1
    assert results.results[0].metadata.get("fruit_type") == img_data[1]["fruit_type"]

@pytest.mark.asyncio
async def test_cohere_multimodal_workflow(multimodal_cohere_config: Tuple[OracleVSMemoryConfig, OracleVSMemoryConfig], img_data: Tuple[List, Dict]) -> None:
    """Test basic query"""
    memory = OracleVSMemory(multimodal_cohere_config[0])
    await memory.clear()

    for doc in img_data[0]:
        await memory.add(
            MemoryContent(
                content= Image.from_pil(PILImage.open(doc["url"])),
                mime_type=MemoryMimeType.IMAGE,
                metadata={"fruit_type": doc["fruit_type"]},
            )
        )

    memory_txt = OracleVSMemory(multimodal_cohere_config[1])

    results = await memory_txt.query("A photo of a strawberry")

    assert len(results.results) == 1
    assert results.results[0].metadata.get("fruit_type") == "strawberry"
