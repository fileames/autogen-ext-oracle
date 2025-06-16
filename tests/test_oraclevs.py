from pathlib import Path

import pytest
import pytest_asyncio
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import UserMessage
from autogen_ext_oracle import OracleVSMemory, OracleVSMemoryConfig

# Skip all tests if oracledb is not available
try:
    import oracledb  # pyright: ignore[reportUnusedImport]
except ImportError:
    pytest.skip("oracledb not available", allow_module_level=True)


@pytest_asyncio.fixture()
async def txt_config() -> OracleVSMemoryConfig:
    """Create base configuration."""

    username = "username"
    password = "password"
    dsn = "dsn"

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

    return config

@pytest_asyncio.fixture()
async def txt_default_config() -> OracleVSMemoryConfig:
    """Create base configuration."""

    username = "onnxuser"
    password = "onnxuser"
    dsn = "100.94.148.194:1527/cdb1_pdb1.regress.rdbms.dev.us.oracle.com"

    connection = await oracledb.connect_async(user=username, password=password, dsn=dsn)

    config=OracleVSMemoryConfig(
        client=connection,
        params = {
            "provider" : "database", 
            "model"    : "allminilm" 
            },
        table_name="mytable",
    )

    return config


@pytest_asyncio.fixture()
async def txt_sync_config() -> OracleVSMemoryConfig:
    """Create base configuration."""

    username = "onnxuser"
    password = "onnxuser"
    dsn = "100.94.148.194:1527/cdb1_pdb1.regress.rdbms.dev.us.oracle.com"

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

    return config

@pytest_asyncio.fixture()
async def txt_pool_config() -> OracleVSMemoryConfig:
    """Create base configuration."""

    username = "onnxuser"
    password = "onnxuser"
    dsn = "100.94.148.194:1527/cdb1_pdb1.regress.rdbms.dev.us.oracle.com"

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


    return (config, config2)


@pytest.mark.asyncio
async def test_basic_workflow(txt_config: OracleVSMemoryConfig) -> None:
    """Test basic memory operations."""
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

    await memory.reset()
    await memory.close()

@pytest.mark.asyncio
async def test_default_workflow(txt_default_config: OracleVSMemoryConfig) -> None:
    """Test basic memory operations."""
    memory = OracleVSMemory(txt_default_config)
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

    await memory.reset()
    await memory.close()

@pytest.mark.asyncio
async def test_hnsw_index(txt_config: OracleVSMemoryConfig) -> None:
    """Test hnsw."""
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

    await memory.reset()
    await memory.close()


@pytest.mark.asyncio
async def test_ivf_index(txt_config: OracleVSMemoryConfig) -> None:
    """Test ivf."""
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

    await memory.reset()
    await memory.close()


@pytest.mark.asyncio
async def test_error_after_close(txt_config: OracleVSMemoryConfig) -> None:
    """Test test_error_after_close."""
    memory = OracleVSMemory(txt_config)
    await memory.clear()

    await memory.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        )
    )

    await memory.reset()
    await memory.close()

    with pytest.raises(oracledb.Error):
        await memory.add(
            MemoryContent(
                content="Istanbul is known for its rich history, diverse culture.",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "city", "country": "Turkiye"},
            )
        )


@pytest.mark.asyncio
async def test_clear(txt_config: OracleVSMemoryConfig) -> None:
    """Test clear."""
    memory = OracleVSMemory(txt_config)
    await memory.clear()

    await memory.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        )
    )

    await memory.clear()

    results = await memory.query("Tell me about Istanbul")
    assert len(results.results) == 0

    await memory.reset()
    await memory.close()

@pytest.mark.asyncio
async def test_reset(txt_config: OracleVSMemoryConfig) -> None:
    """Test reset."""
    memory = OracleVSMemory(txt_config)
    await memory.clear()

    await memory.add(
        MemoryContent(
            content="Paris is known for the Eiffel Tower and amazing cuisine.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "city", "country": "France"},
        )
    )

    await memory.reset()

    with pytest.raises(oracledb.Error):
        results = await memory.query("Tell me about Istanbul")

    await memory.close()


@pytest.mark.asyncio
async def test_close_conn(txt_config: OracleVSMemoryConfig) -> None:
    """Test reset."""
    await txt_config.client.close()

    memory = OracleVSMemory(txt_config)

    with pytest.raises(oracledb.InterfaceError, match="not connected to database"):
        await memory.add(
            MemoryContent(
                content="Paris is known for the Eiffel Tower and amazing cuisine.",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "city", "country": "France"},
            )
        )

txt_sync_config
    
@pytest.mark.asyncio
async def test_sync_conn(txt_sync_config: OracleVSMemoryConfig) -> None:
    """Test reset."""
    memory = OracleVSMemory(txt_sync_config)

    with pytest.raises(TypeError, match="oracledb.AsyncConnection or oracledb.AsyncConnectionPool"):
        await memory.add(
            MemoryContent(
                content="Paris is known for the Eiffel Tower and amazing cuisine.",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "city", "country": "France"},
            )
        )


@pytest.mark.asyncio
async def test_content_types(txt_config: OracleVSMemoryConfig) -> None:
    """Test reset."""
    memory = OracleVSMemory(txt_config)
    await memory.clear()

    with pytest.raises(ValueError): #, match="Unsupported content type"):
        await memory.add(MemoryContent(content=b"binary data", mime_type=MemoryMimeType.BINARY))

    with pytest.raises(ValueError):
        await memory.add(MemoryContent(content="not a dict", mime_type=MemoryMimeType.JSON))

    # TODO: what to do with images
    await memory.close()

@pytest.mark.asyncio
async def test_metadata_handling(txt_config: OracleVSMemoryConfig) -> None:
    """Test metadata handling with default threshold."""
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

    await memory.close()

@pytest.mark.asyncio
async def test_pool(txt_pool_config: OracleVSMemoryConfig) -> None:
    """Test reset."""
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
    
@pytest.mark.asyncio
async def test_model_context_update(txt_config: OracleVSMemoryConfig) -> None:
    """Test updating model context with retrieved memories."""
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

    await memory.close()

@pytest.mark.asyncio
async def test_filter(txt_config: OracleVSMemoryConfig) -> None:
    """Test basic memory operations."""
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

    await memory.reset()
    await memory.close()


