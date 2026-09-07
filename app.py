# ============================================================
# Streamlit - MongoDB Lab Test Vector Search
#
# Database  : lab_db
# Collection: test_catalog
#
# Function:
# 1. Connect MongoDB Atlas
# 2. Load Sentence Transformer
# 3. Create embedding text
# 4. Create vectors
# 5. Save vectors to MongoDB
# 6. Create / check MongoDB Vector Search Index
# 7. MongoDB Atlas Vector Search
# 8. Semantic Search
# 9. Preview vector data
# ============================================================


# ============================================================
# 0. Imports
# ============================================================

import time

import streamlit as st
import pandas as pd

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Lab Test Vector Search",
    page_icon="🧬",
    layout="wide",
)


# ============================================================
# 2. Configuration
# ============================================================

MONGODB_URI = st.secrets.get(
    "MONGODB_URI",
    "",
)

DATABASE_NAME = "lab_db"

COLLECTION_NAME = "test_catalog"


# ============================================================
# 3. Embedding Model
# ============================================================

MODEL_NAME = (
    "paraphrase-multilingual-MiniLM-L12-v2"
)

VECTOR_FIELD = "embedding"

TEXT_FIELD = "embedding_text"

MODEL_FIELD = "embedding_model"

DIMENSION_FIELD = "embedding_dimensions"


# ============================================================
# 4. Vector Search Configuration
# ============================================================

VECTOR_INDEX_NAME = "vector_index"

VECTOR_SEARCH_LIMIT = 10

VECTOR_SEARCH_CANDIDATES = 100

EXPECTED_VECTOR_DIMENSIONS = 384


# ============================================================
# 5. Embedding Fields
# ============================================================

EMBEDDING_FIELDS = [
    "ชื่อการทดสอบ",
    "วิธีการตรวจวิเคราะห์",
    "สิ่งส่งตรวจ/ปริมาตร (ml) Preservative",
    "การนำส่งและข้อควรระวัง",
    "ค่าอ้างอิง sensitivity/Detection Range",
    "ข้อบ่งชี้การทดสอบ",
    "รายงานผลปกติ",
]


# ============================================================
# 6. Header
# ============================================================

st.title(
    "🧬 Lab Test Vector Search"
)

st.markdown(
    """
    เครื่องมือสำหรับค้นหาข้อมูลการตรวจทางห้องปฏิบัติการ
    ด้วย **Semantic Vector Search**

    ระบบจะนำคำค้นของผู้ใช้ไปสร้าง Vector
    แล้วค้นหาข้อมูลที่มีความหมายใกล้เคียงกัน
    จาก MongoDB Atlas
    """
)


# ============================================================
# 7. Sidebar
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Configuration"
    )

    st.write(
        "Database"
    )

    st.code(
        DATABASE_NAME
    )

    st.write(
        "Collection"
    )

    st.code(
        COLLECTION_NAME
    )

    st.write(
        "Embedding Model"
    )

    st.code(
        MODEL_NAME
    )

    st.write(
        "Vector Index"
    )

    st.code(
        VECTOR_INDEX_NAME
    )

    st.write(
        "Vector Field"
    )

    st.code(
        VECTOR_FIELD
    )

    st.write(
        "Vector Dimensions"
    )

    st.info(
        f"{EXPECTED_VECTOR_DIMENSIONS} dimensions"
    )

    st.divider()

    st.subheader(
        "Fields used for Vector"
    )

    for field in EMBEDDING_FIELDS:

        st.write(
            f"• {field}"
        )

    st.divider()

    st.subheader(
        "Vector Search"
    )

    st.write(
        f"Top Results: {VECTOR_SEARCH_LIMIT}"
    )

    st.write(
        f"Candidates: {VECTOR_SEARCH_CANDIDATES}"
    )


# ============================================================
# 8. Check MongoDB URI
# ============================================================

if not MONGODB_URI:

    st.error(
        "ไม่พบ MONGODB_URI"
    )

    st.info(
        """
        กรุณาสร้างไฟล์:

        `.streamlit/secrets.toml`

        แล้วใส่:

        MONGODB_URI = "your-mongodb-connection-string"
        """
    )

    st.stop()


# ============================================================
# 9. Connect MongoDB
# ============================================================

@st.cache_resource
def get_mongodb_collection():

    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=10000,
    )

    client.admin.command(
        "ping"
    )

    db = client[
        DATABASE_NAME
    ]

    collection = db[
        COLLECTION_NAME
    ]

    return client, collection


# ============================================================
# 10. Load Embedding Model
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        MODEL_NAME
    )

    return model


# ============================================================
# 11. Connect MongoDB
# ============================================================

try:

    client, collection = (
        get_mongodb_collection()
    )

    st.success(
        "🟢 MongoDB Atlas connection: OK"
    )

except Exception as e:

    st.error(
        "ไม่สามารถเชื่อมต่อ MongoDB Atlas ได้"
    )

    st.exception(
        e
    )

    st.stop()


# ============================================================
# 12. Load Embedding Model
# ============================================================

try:

    with st.spinner(
        "กำลังโหลด Embedding Model..."
    ):

        model = load_embedding_model()

except Exception as e:

    st.error(
        "ไม่สามารถโหลด Embedding Model ได้"
    )

    st.exception(
        e
    )

    st.stop()


# ============================================================
# 13. Model Information
# ============================================================

try:

    vector_dimension = (
        model.get_sentence_embedding_dimension()
    )

except Exception:

    vector_dimension = EXPECTED_VECTOR_DIMENSIONS


# ============================================================
# 14. Helper Functions
# ============================================================

def safe_text(value):

    if value is None:

        return ""

    try:

        if pd.isna(value):

            return ""

    except Exception:

        pass

    return str(
        value
    ).strip()


# ============================================================
# 15. Create Embedding Text
# ============================================================

def create_embedding_text(
    document
):

    parts = []

    for field in EMBEDDING_FIELDS:

        value = safe_text(
            document.get(field)
        )

        if value:

            parts.append(
                f"{field}: {value}"
            )

    return "\n".join(
        parts
    )


# ============================================================
# 16. Get Vector Search Index
# ============================================================

def get_vector_search_index():

    try:

        indexes = list(
            collection.list_search_indexes(
                VECTOR_INDEX_NAME
            )
        )

        if not indexes:

            return None, None

        return indexes[0], None

    except Exception as e:

        return None, e


# ============================================================
# 17. Create Vector Search Index
# ============================================================

def create_vector_search_index():

    index_definition = {
        "fields": [
            {
                "type": "vector",
                "path": VECTOR_FIELD,
                "numDimensions": EXPECTED_VECTOR_DIMENSIONS,
                "similarity": "cosine",
            }
        ]
    }

    search_index_model = SearchIndexModel(
        definition=index_definition,
        name=VECTOR_INDEX_NAME,
        type="vectorSearch",
    )

    result = (
        collection.create_search_index(
            search_index_model
        )
    )

    return result


# ============================================================
# 18. Check Vector Search Index
# ============================================================

vector_index, vector_index_error = (
    get_vector_search_index()
)


# ============================================================
# 19. Vector Search Index Status
# ============================================================

st.divider()

st.subheader(
    "🗂️ Vector Search Index"
)


if vector_index_error:

    st.error(
        "ไม่สามารถตรวจสอบ Vector Search Index ได้"
    )

    st.code(
        str(
            vector_index_error
        )
    )

    st.info(
        """
        ตรวจสอบสิทธิ์ของ MongoDB User
        ว่าสามารถอ่าน Search Index ได้
        """
    )


elif vector_index is None:

    st.warning(
        f"""
        🟡 ยังไม่พบ Vector Search Index

        Index ที่ระบบต้องการ:

        `{VECTOR_INDEX_NAME}`
        """
    )

    st.code(
        """
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
        """,
        language="json",
    )

    if st.button(
        "🛠️ สร้าง Vector Search Index",
        type="primary",
        width="stretch",
    ):

        try:

            with st.spinner(
                "กำลังสร้าง Vector Search Index..."
            ):

                create_vector_search_index()

            st.success(
                f"""
                สร้าง Index `{VECTOR_INDEX_NAME}`
                เรียบร้อยแล้ว

                MongoDB กำลังสร้าง Index
                กรุณารอสักครู่
                """
            )

            time.sleep(2)

            st.rerun()

        except Exception as e:

            st.error(
                "ไม่สามารถสร้าง Vector Search Index ได้"
            )

            st.exception(
                e
            )

            st.info(
                """
                ถ้ามี Index ชื่อนี้อยู่แล้ว
                แต่อ่านสถานะไม่ได้
                ให้ตรวจสอบ Index ใน MongoDB Atlas
                """
            )


else:

    index_status = vector_index.get(
        "status",
        "UNKNOWN",
    )

    index_queryable = vector_index.get(
        "queryable",
        False,
    )

    index_type = vector_index.get(
        "type",
        "unknown",
    )

    index_definition = vector_index.get(
        "latestDefinition",
        {},
    )

    status_col1, status_col2, status_col3 = (
        st.columns(3)
    )

    with status_col1:

        st.metric(
            "Index",
            VECTOR_INDEX_NAME,
        )

    with status_col2:

        st.metric(
            "Status",
            index_status,
        )

    with status_col3:

        st.metric(
            "Queryable",
            "YES"
            if index_queryable
            else "NO",
        )

    if (
        index_status == "READY"
        and index_queryable
    ):

        st.success(
            "🟢 Vector Search Index พร้อมใช้งาน"
        )

    elif index_status == "BUILDING":

        st.warning(
            """
            🟡 MongoDB กำลังสร้าง Vector Search Index

            กรุณารอจนสถานะเป็น READY
            """
        )

    elif index_status == "FAILED":

        st.error(
            """
            🔴 Vector Search Index FAILED

            กรุณาตรวจสอบ definition ของ Index
            ใน MongoDB Atlas
            """
        )

    else:

        st.warning(
            f"""
            🟡 Vector Search Index ยังไม่พร้อม

            Status:
            {index_status}

            Queryable:
            {index_queryable}
            """
        )

    with st.expander(
        "🔍 Index Definition",
        expanded=False,
    ):

        st.json(
            index_definition
        )

    st.caption(
        f"Index Type: {index_type}"
    )


# ============================================================
# 20. Database Statistics
# ============================================================

total_documents = (
    collection.count_documents({})
)

vector_documents = (
    collection.count_documents(
        {
            VECTOR_FIELD: {
                "$exists": True
            }
        }
    )
)

remaining_documents = (
    total_documents
    - vector_documents
)


# ============================================================
# 21. Database Status
# ============================================================

st.divider()

st.subheader(
    "📊 Database Status"
)

col1, col2, col3, col4 = (
    st.columns(4)
)

with col1:

    st.metric(
        "Documents",
        total_documents,
    )

with col2:

    st.metric(
        "มี Vector แล้ว",
        vector_documents,
    )

with col3:

    st.metric(
        "ยังไม่มี Vector",
        remaining_documents,
    )

with col4:

    st.metric(
        "Vector Dimensions",
        vector_dimension,
    )


# ============================================================
# 22. Vector Data Validation
# ============================================================

invalid_vector_documents = 0

sample_vectors = collection.find(
    {
        VECTOR_FIELD: {
            "$exists": True
        }
    },
    {
        VECTOR_FIELD: 1
    },
).limit(100)

for item in sample_vectors:

    vector = item.get(
        VECTOR_FIELD,
        [],
    )

    if (
        not isinstance(
            vector,
            list,
        )
        or len(vector)
        != EXPECTED_VECTOR_DIMENSIONS
    ):

        invalid_vector_documents += 1


if vector_documents > 0:

    if invalid_vector_documents == 0:

        st.success(
            f"""
            🟢 ตรวจสอบ Vector เบื้องต้นผ่าน

            พบ Vector:
            {vector_documents:,} รายการ

            Dimensions:
            {EXPECTED_VECTOR_DIMENSIONS}
            """
        )

    else:

        st.warning(
            f"""
            🟡 พบ Vector ที่อาจมีปัญหา
            จากตัวอย่างที่ตรวจสอบ:

            {invalid_vector_documents:,} รายการ
            """
        )


# ============================================================
# 23. Semantic Vector Search
# ============================================================

st.divider()

st.subheader(
    "🔎 Semantic Vector Search"
)

st.caption(
    """
    ค้นหาจากความหมายของคำค้น
    ไม่จำเป็นต้องตรงกับข้อความในฐานข้อมูลแบบคำต่อคำ
    """
)


# ============================================================
# 24. Search Input
# ============================================================

search_text = st.text_input(
    "ค้นหา Lab Test",
    placeholder=(
        "เช่น ตรวจหาโรคซิฟิลิส, "
        "ตรวจ antibody, "
        "ตรวจโรคติดเชื้อจากเลือด..."
    ),
    type="default",
)


# ============================================================
# 25. Search Settings
# ============================================================

search_col1, search_col2 = (
    st.columns(2)
)

with search_col1:

    result_limit = st.slider(
        "จำนวนผลลัพธ์",
        min_value=1,
        max_value=20,
        value=10,
        step=1,
    )

with search_col2:

    candidate_count = st.slider(
        "จำนวน Candidates",
        min_value=20,
        max_value=500,
        value=100,
        step=10,
    )


# ============================================================
# 26. Vector Search Function
# ============================================================

def vector_search(
    query_text,
    limit=10,
    num_candidates=100,
):

    # --------------------------------------------------------
    # Create Query Vector
    # --------------------------------------------------------

    query_vector = model.encode(
        query_text,
        normalize_embeddings=True,
    )

    query_vector = (
        query_vector.tolist()
    )

    # --------------------------------------------------------
    # Validate Query Vector
    # --------------------------------------------------------

    if len(query_vector) != EXPECTED_VECTOR_DIMENSIONS:

        raise ValueError(
            f"""
            Query Vector มีขนาด
            {len(query_vector)}

            แต่ระบบต้องการ
            {EXPECTED_VECTOR_DIMENSIONS}
            """
        )

    # --------------------------------------------------------
    # MongoDB Vector Search
    # --------------------------------------------------------

    pipeline = [

        {
            "$vectorSearch": {

                "index":
                    VECTOR_INDEX_NAME,

                "path":
                    VECTOR_FIELD,

                "queryVector":
                    query_vector,

                "numCandidates":
                    num_candidates,

                "limit":
                    limit,
            }
        },

        {
            "$project": {

                "_id": 0,

                "รหัส": 1,

                "ชื่อการทดสอบ": 1,

                "วิธีการตรวจวิเคราะห์": 1,

                "สิ่งส่งตรวจ/ปริมาตร (ml) Preservative": 1,

                "การนำส่งและข้อควรระวัง": 1,

                "ค่าอ้างอิง sensitivity/Detection Range": 1,

                "ข้อบ่งชี้การทดสอบ": 1,

                "รายงานผลปกติ": 1,

                TEXT_FIELD: 1,

                MODEL_FIELD: 1,

                DIMENSION_FIELD: 1,

                VECTOR_FIELD: 1,

                "score": {
                    "$meta":
                        "vectorSearchScore",
                },
            }
        },
    ]

    results = list(
        collection.aggregate(
            pipeline
        )
    )

    return (
        results,
        query_vector,
    )


# ============================================================
# 27. Execute Vector Search
# ============================================================

search_documents = []

query_vector = []

if search_text.strip():

    keyword = (
        search_text
        .strip()
    )

    # --------------------------------------------------------
    # Check Vector Data
    # --------------------------------------------------------

    if vector_documents == 0:

        st.error(
            """
            ❌ ยังไม่มี Vector ใน MongoDB

            กรุณากด
            "เริ่มสร้าง Vector"
            ก่อน
            """
        )

    # --------------------------------------------------------
    # Check Index
    # --------------------------------------------------------

    elif (
        vector_index is None
    ):

        st.error(
            f"""
            ❌ ไม่พบ Vector Search Index

            Required Index:

            `{VECTOR_INDEX_NAME}`
            """
        )

    # --------------------------------------------------------
    # Check Index Queryable
    # --------------------------------------------------------

    elif not vector_index.get(
        "queryable",
        False,
    ):

        st.warning(
            f"""
            🟡 Vector Search Index ยังไม่พร้อม

            Status:
            {vector_index.get(
                "status",
                "UNKNOWN"
            )}

            Queryable:
            {vector_index.get(
                "queryable",
                False
            )}

            กรุณารอจน Index เป็น READY
            """
        )

    # --------------------------------------------------------
    # Execute Search
    # --------------------------------------------------------

    else:

        with st.spinner(
            "กำลังค้นหาด้วย Vector Search..."
        ):

            try:

                (
                    search_documents,
                    query_vector,
                ) = vector_search(
                    keyword,
                    limit=result_limit,
                    num_candidates=candidate_count,
                )

                # ------------------------------------------------
                # Search Summary
                # ------------------------------------------------

                st.success(
                    f"""
                    🔎 ค้นหา:

                    **{keyword}**

                    พบผลลัพธ์:
                    **{len(search_documents):,} รายการ**
                    """
                )

                # ------------------------------------------------
                # Query Vector
                # ------------------------------------------------

                with st.expander(
                    "🧠 Query Vector",
                    expanded=False,
                ):

                    st.write(
                        "Query:"
                    )

                    st.code(
                        keyword
                    )

                    st.write(
                        "Dimensions:"
                    )

                    st.code(
                        str(
                            len(
                                query_vector
                            )
                        )
                    )

                    st.write(
                        "Vector Preview:"
                    )

                    st.code(
                        str(
                            query_vector[:20]
                        )
                        + " ..."
                    )

                # ------------------------------------------------
                # Search Results
                # ------------------------------------------------

                if search_documents:

                    st.subheader(
                        "📊 Search Results"
                    )

                    search_rows = []

                    for rank, document in enumerate(
                        search_documents,
                        start=1,
                    ):

                        score = float(
                            document.get(
                                "score",
                                0,
                            )
                        )

                        search_rows.append(
                            {
                                "อันดับ":
                                    rank,

                                "Similarity":
                                    round(
                                        score,
                                        4,
                                    ),

                                "รหัส":
                                    document.get(
                                        "รหัส",
                                        "",
                                    ),

                                "ชื่อการทดสอบ":
                                    document.get(
                                        "ชื่อการทดสอบ",
                                        "",
                                    ),

                                "ข้อบ่งชี้":
                                    document.get(
                                        "ข้อบ่งชี้การทดสอบ",
                                        "",
                                    ),

                                "วิธีการตรวจ":
                                    document.get(
                                        "วิธีการตรวจวิเคราะห์",
                                        "",
                                    ),
                            }
                        )

                    search_df = pd.DataFrame(
                        search_rows
                    )

                    st.dataframe(
                        search_df,
                        width="stretch",
                        hide_index=True,
                    )

                else:

                    st.warning(
                        """
                        ไม่พบผลลัพธ์จาก Vector Search

                        ถ้า Vector มีครบและ Index
                        เป็น READY แล้ว
                        เราสามารถตรวจสอบข้อมูล Vector
                        และ Index Definition ต่อได้
                        """
                    )

            except Exception as e:

                st.error(
                    "เกิดข้อผิดพลาดในการทำ Vector Search"
                )

                st.exception(
                    e
                )


# ============================================================
# 28. Selected Search Result
# ============================================================

if search_documents:

    st.divider()

    st.subheader(
        "📄 รายละเอียดผลการค้นหา"
    )

    selected_index = st.selectbox(
        "เลือกรายการ",
        range(
            len(search_documents)
        ),
        format_func=lambda index: (
            f"{search_documents[index].get('รหัส', '')} - "
            f"{search_documents[index].get('ชื่อการทดสอบ', '')} "
            f"(score: "
            f"{float(search_documents[index].get('score', 0)):.4f})"
        ),
    )

    selected_document = (
        search_documents[
            selected_index
        ]
    )

    # --------------------------------------------------------
    # Similarity Score
    # --------------------------------------------------------

    score = float(
        selected_document.get(
            "score",
            0,
        )
    )

    score_col1, score_col2, score_col3 = (
        st.columns(3)
    )

    with score_col1:

        st.metric(
            "Similarity Score",
            f"{score:.4f}",
        )

    with score_col2:

        st.metric(
            "Vector Dimensions",
            selected_document.get(
                DIMENSION_FIELD,
                vector_dimension,
            ),
        )

    with score_col3:

        st.metric(
            "Embedding Model",
            selected_document.get(
                MODEL_FIELD,
                MODEL_NAME,
            ),
        )

    # --------------------------------------------------------
    # Basic Information
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "### รหัส"
        )

        st.write(
            selected_document.get(
                "รหัส",
                "-",
            )
        )

    with col2:

        st.write(
            "### ชื่อการทดสอบ"
        )

        st.write(
            selected_document.get(
                "ชื่อการทดสอบ",
                "-",
            )
        )

    # --------------------------------------------------------
    # Test Information
    # --------------------------------------------------------

    with st.expander(
        "🔬 ข้อมูลการตรวจ",
        expanded=True,
    ):

        for field in [
            "วิธีการตรวจวิเคราะห์",
            "สิ่งส่งตรวจ/ปริมาตร (ml) Preservative",
            "การนำส่งและข้อควรระวัง",
            "ค่าอ้างอิง sensitivity/Detection Range",
            "ข้อบ่งชี้การทดสอบ",
            "รายงานผลปกติ",
        ]:

            st.markdown(
                f"**{field}**"
            )

            value = safe_text(
                selected_document.get(
                    field
                )
            )

            if value:

                st.write(
                    value
                )

            else:

                st.caption(
                    "-"
                )

    # --------------------------------------------------------
    # Embedding Text
    # --------------------------------------------------------

    with st.expander(
        "📝 Embedding Text",
        expanded=False,
    ):

        embedding_text = (
            selected_document.get(
                TEXT_FIELD,
                "",
            )
        )

        if embedding_text:

            st.text_area(
                "Embedding Text",
                value=embedding_text,
                height=250,
                label_visibility="collapsed",
            )

        else:

            st.warning(
                "รายการนี้ยังไม่มี Embedding Text"
            )

    # --------------------------------------------------------
    # Vector
    # --------------------------------------------------------

    with st.expander(
        "🔢 Vector",
        expanded=False,
    ):

        vector = (
            selected_document.get(
                VECTOR_FIELD,
                [],
            )
        )

        if vector:

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:

                st.metric(
                    "Dimensions",
                    len(vector),
                )

            with col2:

                st.metric(
                    "Model",
                    selected_document.get(
                        MODEL_FIELD,
                        "-",
                    ),
                )

            with col3:

                st.metric(
                    "Vector Size",
                    f"{len(vector)} numbers",
                )

            st.write(
                "Vector ตัวอย่าง 20 ค่าแรก"
            )

            st.code(
                str(
                    vector[:20]
                )
            )

        else:

            st.info(
                "รายการนี้ยังไม่มี Vector"
            )


# ============================================================
# 29. Preview Data
# ============================================================

st.divider()

st.subheader(
    "👀 Preview Data"
)

preview_documents = list(
    collection.find(
        {},
        {
            "_id": 0,

            "รหัส": 1,

            "ชื่อการทดสอบ": 1,

            "ข้อบ่งชี้การทดสอบ": 1,

            VECTOR_FIELD: 1,
        },
    )
    .sort(
        "รหัส",
        1,
    )
    .limit(10)
)


if preview_documents:

    preview_rows = []

    for item in preview_documents:

        vector = item.get(
            VECTOR_FIELD,
            [],
        )

        preview_rows.append(
            {
                "รหัส":
                    item.get(
                        "รหัส",
                    ),

                "ชื่อการทดสอบ":
                    item.get(
                        "ชื่อการทดสอบ",
                    ),

                "ข้อบ่งชี้การทดสอบ":
                    item.get(
                        "ข้อบ่งชี้การทดสอบ",
                    ),

                "มี Vector":
                    (
                        "✓"
                        if vector
                        else "—"
                    ),
            }
        )

    preview_df = pd.DataFrame(
        preview_rows
    )

    st.dataframe(
        preview_df,
        width="stretch",
        hide_index=True,
    )

else:

    st.warning(
        "ไม่พบข้อมูลใน collection"
    )


# ============================================================
# 30. Preview Embedding Text
# ============================================================

st.subheader(
    "📝 ตัวอย่างข้อความที่จะนำไปสร้าง Vector"
)

sample_document = collection.find_one(
    {},
    {
        "_id": 0
    },
)

if sample_document:

    sample_text = (
        create_embedding_text(
            sample_document
        )
    )

    st.text_area(
        "Embedding Text",
        value=sample_text,
        height=250,
    )


# ============================================================
# 31. Vectorize Options
# ============================================================

st.divider()

st.subheader(
    "🚀 สร้าง Vector"
)

option = st.radio(
    "เลือกข้อมูลที่จะสร้าง Vector",
    [
        "เฉพาะข้อมูลที่ยังไม่มี Vector",
        "สร้างใหม่ทั้งหมด",
    ],
    horizontal=True,
)


if (
    option
    == "เฉพาะข้อมูลที่ยังไม่มี Vector"
):

    vectorize_query = {
        VECTOR_FIELD: {
            "$exists": False
        }
    }

else:

    vectorize_query = {}


# ============================================================
# 32. Count Target Documents
# ============================================================

target_count = (
    collection.count_documents(
        vectorize_query
    )
)

st.info(
    f"""
    ข้อมูลที่จะประมวลผล:
    **{target_count:,} รายการ**
    """
)


# ============================================================
# 33. Generate Vector Button
# ============================================================

if st.button(
    "🧬 เริ่มสร้าง Vector",
    type="primary",
    width="stretch",
):

    if target_count == 0:

        st.success(
            "ไม่มีข้อมูลที่ต้องสร้าง Vector"
        )

    else:

        progress = st.progress(
            0,
            text="กำลังเตรียมข้อมูล...",
        )

        status_text = st.empty()

        success_count = 0

        error_count = 0

        errors = []

        # ----------------------------------------------------
        # Read target documents
        # ----------------------------------------------------

        documents = list(
            collection.find(
                vectorize_query,
                {
                    "_id": 1,

                    "รหัส": 1,

                    "ชื่อการทดสอบ": 1,

                    "วิธีการตรวจวิเคราะห์": 1,

                    "สิ่งส่งตรวจ/ปริมาตร (ml) Preservative": 1,

                    "การนำส่งและข้อควรระวัง": 1,

                    "ค่าอ้างอิง sensitivity/Detection Range": 1,

                    "ข้อบ่งชี้การทดสอบ": 1,

                    "รายงานผลปกติ": 1,
                },
            )
            .sort(
                "รหัส",
                1,
            )
        )

        # ----------------------------------------------------
        # Process
        # ----------------------------------------------------

        for index, document in enumerate(
            documents,
            start=1,
        ):

            try:

                test_code = document.get(
                    "รหัส",
                    "",
                )

                test_name = document.get(
                    "ชื่อการทดสอบ",
                    "",
                )

                status_text.write(
                    f"""
                    กำลังประมวลผล
                    **{index:,} / {target_count:,}**

                    รหัส: `{test_code}`

                    {test_name}
                    """
                )

                # ------------------------------------------------
                # Create embedding text
                # ------------------------------------------------

                embedding_text = (
                    create_embedding_text(
                        document
                    )
                )

                if not embedding_text:

                    raise ValueError(
                        "ไม่มีข้อมูลสำหรับสร้าง embedding"
                    )

                # ------------------------------------------------
                # Generate embedding
                # ------------------------------------------------

                vector = model.encode(
                    embedding_text,
                    normalize_embeddings=True,
                )

                # ------------------------------------------------
                # Convert numpy -> list
                # ------------------------------------------------

                vector_list = (
                    vector.tolist()
                )

                # ------------------------------------------------
                # Validate vector
                # ------------------------------------------------

                if len(vector_list) != (
                    EXPECTED_VECTOR_DIMENSIONS
                ):

                    raise ValueError(
                        f"""
                        Vector dimensions ไม่ถูกต้อง

                        ได้:
                        {len(vector_list)}

                        ต้องเป็น:
                        {EXPECTED_VECTOR_DIMENSIONS}
                        """
                    )

                # ------------------------------------------------
                # Update MongoDB
                # ------------------------------------------------

                collection.update_one(
                    {
                        "_id":
                            document["_id"]
                    },
                    {
                        "$set": {

                            TEXT_FIELD:
                                embedding_text,

                            VECTOR_FIELD:
                                vector_list,

                            MODEL_FIELD:
                                MODEL_NAME,

                            DIMENSION_FIELD:
                                vector_dimension,
                        }
                    },
                )

                success_count += 1

            except Exception as e:

                error_count += 1

                errors.append(
                    {
                        "รหัส":
                            document.get(
                                "รหัส"
                            ),

                        "ชื่อการทดสอบ":
                            document.get(
                                "ชื่อการทดสอบ"
                            ),

                        "error":
                            str(e),
                    }
                )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            percent = (
                index / target_count
            )

            progress.progress(
                percent,
                text=(
                    f"Progress: "
                    f"{index:,}/{target_count:,}"
                ),
            )

        # ----------------------------------------------------
        # Finish
        # ----------------------------------------------------

        progress.empty()

        status_text.empty()

        st.success(
            f"""
            สร้าง Vector เสร็จแล้ว

            สำเร็จ: {success_count:,}

            ผิดพลาด: {error_count:,}

            รวม: {target_count:,}
            """
        )

        # ----------------------------------------------------
        # Errors
        # ----------------------------------------------------

        if errors:

            st.subheader(
                "⚠️ รายการที่เกิดข้อผิดพลาด"
            )

            error_df = pd.DataFrame(
                errors
            )

            st.dataframe(
                error_df,
                width="stretch",
                hide_index=True,
            )

        st.rerun()


# ============================================================
# 34. Vector Data
# ============================================================

st.divider()

st.subheader(
    "🔢 Vector Data"
)

vector_sample = list(
    collection.find(
        {
            VECTOR_FIELD: {
                "$exists": True
            }
        },
        {
            "_id": 0,

            "รหัส": 1,

            "ชื่อการทดสอบ": 1,

            TEXT_FIELD: 1,

            MODEL_FIELD: 1,

            DIMENSION_FIELD: 1,

            VECTOR_FIELD: 1,
        },
    )
    .sort(
        "รหัส",
        1,
    )
    .limit(10)
)


if vector_sample:

    vector_preview_rows = []

    for item in vector_sample:

        vector = item.get(
            VECTOR_FIELD,
            [],
        )

        vector_preview_rows.append(
            {
                "รหัส":
                    item.get(
                        "รหัส",
                    ),

                "ชื่อการทดสอบ":
                    item.get(
                        "ชื่อการทดสอบ",
                    ),

                "Model":
                    item.get(
                        MODEL_FIELD,
                    ),

                "Dimensions":
                    item.get(
                        DIMENSION_FIELD,
                    ),

                "Vector Preview":
                    (
                        str(
                            vector[:10]
                        )
                        + " ..."
                    ),
            }
        )

    vector_preview_df = (
        pd.DataFrame(
            vector_preview_rows
        )
    )

    st.dataframe(
        vector_preview_df,
        width="stretch",
        hide_index=True,
    )

else:

    st.info(
        "ยังไม่มี Vector ใน collection"
    )


# ============================================================
# 35. Refresh
# ============================================================

st.divider()

if st.button(
    "🔄 Refresh",
    width="stretch",
):

    st.rerun()


# ============================================================
# 36. Footer
# ============================================================

st.divider()

st.caption(
    "Lab Test Vector Search • "
    "MongoDB Atlas • "
    "Sentence Transformers"
)