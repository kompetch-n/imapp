# ============================================================
# Streamlit - MongoDB Lab Test Vectorizer
#
# Database  : lab_db
# Collection: test_catalog
#
# Function:
# 1. Connect MongoDB Atlas
# 2. Read lab test data
# 3. Search lab test data
# 4. Create embedding text
# 5. Convert text -> vector
# 6. Save vector back to MongoDB
# 7. Preview vector data
# ============================================================

import streamlit as st
import pandas as pd

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Lab Test Vectorizer",
    page_icon="🧬",
    layout="wide",
)


# ============================================================
# 2. Configuration
# ============================================================

MONGODB_URI = st.secrets.get(
    "MONGODB_URI",
    ""
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
# 4. Embedding Fields
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
# 5. Search Fields
# ============================================================

SEARCH_FIELDS = [
    "รหัส",
    "ชื่อการทดสอบ",
    "วิธีการตรวจวิเคราะห์",
    "สิ่งส่งตรวจ/ปริมาตร (ml) Preservative",
    "การนำส่งและข้อควรระวัง",
    "ค่าอ้างอิง sensitivity/Detection Range",
    "ข้อบ่งชี้การทดสอบ",
    "รายงานผลปกติ",
    TEXT_FIELD,
]


# ============================================================
# 6. Header
# ============================================================

st.title(
    "🧬 Lab Test Vectorizer"
)

st.markdown(
    """
    เครื่องมือสำหรับจัดการข้อมูลการตรวจทางห้องปฏิบัติการ
    จาก MongoDB Atlas

    สามารถค้นหาข้อมูล สร้าง Embedding และเก็บ Vector
    กลับเข้า collection `test_catalog`
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
        "Vector Dimensions"
    )

    st.info(
        "384 dimensions"
    )

    st.divider()

    st.subheader(
        "Fields used for Vector"
    )

    for field in EMBEDDING_FIELDS:

        st.write(
            f"• {field}"
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

        `MONGODB_URI = "your-mongodb-connection-string"`
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
# 11. Connect
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
# 12. Load Model
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

    vector_dimension = 384


# ============================================================
# 14. Get MongoDB Statistics
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
# 15. Statistics
# ============================================================

st.subheader(
    "📊 Database Status"
)

col1, col2, col3, col4 = (
    st.columns(4)
)

with col1:

    st.metric(
        "Documents",
        total_documents
    )

with col2:

    st.metric(
        "มี Vector แล้ว",
        vector_documents
    )

with col3:

    st.metric(
        "ยังไม่มี Vector",
        remaining_documents
    )

with col4:

    st.metric(
        "Vector Dimensions",
        vector_dimension
    )


# ============================================================
# 16. Create Embedding Text
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
# 17. Search
# ============================================================

st.divider()

st.subheader(
    "🔎 ค้นหา Lab Test"
)

st.caption(
    "ค้นหาได้จากรหัส ชื่อการทดสอบ วิธีตรวจ สิ่งส่งตรวจ "
    "ข้อบ่งชี้ และข้อมูลที่ใช้สร้าง Vector"
)

search_text = st.text_input(
    "ค้นหา",
    placeholder="เช่น RPR, TPHA, ซิฟิลิส, ตรวจเลือด, antibody...",
    type="default",
    label_visibility="collapsed",
)


# ============================================================
# 18. Build Search Query
# ============================================================

if search_text.strip():

    keyword = (
        search_text
        .strip()
    )

    regex_pattern = keyword

    search_conditions = []

    for field in SEARCH_FIELDS:

        search_conditions.append(
            {
                field: {
                    "$regex": regex_pattern,
                    "$options": "i",
                }
            }
        )

    search_query = {
        "$or": search_conditions
    }

else:

    search_query = {}


# ============================================================
# 19. Search Result Count
# ============================================================

search_result_count = (
    collection.count_documents(
        search_query
    )
)

st.write(
    f"พบข้อมูล **{search_result_count:,} รายการ**"
)


# ============================================================
# 20. Search Result
# ============================================================

search_documents = list(
    collection.find(
        search_query,
        {
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
        },
    )
    .sort(
        "รหัส",
        1
    )
)


# ============================================================
# 21. Search Result Table
# ============================================================

if search_documents:

    search_rows = []

    for document in search_documents:

        vector = document.get(
            VECTOR_FIELD,
            []
        )

        search_rows.append(
            {
                "รหัส": document.get(
                    "รหัส",
                    ""
                ),

                "ชื่อการทดสอบ": document.get(
                    "ชื่อการทดสอบ",
                    ""
                ),

                "วิธีการตรวจวิเคราะห์": document.get(
                    "วิธีการตรวจวิเคราะห์",
                    ""
                ),

                "สิ่งส่งตรวจ": document.get(
                    "สิ่งส่งตรวจ/ปริมาตร (ml) Preservative",
                    ""
                ),

                "ข้อบ่งชี้": document.get(
                    "ข้อบ่งชี้การทดสอบ",
                    ""
                ),

                "มี Vector": (
                    "✓"
                    if vector
                    else "—"
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

    if search_text.strip():

        st.warning(
            "ไม่พบข้อมูลที่ตรงกับคำค้นหา"
        )

    else:

        st.info(
            "กรุณาพิมพ์คำค้นหา"
        )


# ============================================================
# 22. Selected Search Result
# ============================================================

if search_documents:

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
            f"{search_documents[index].get('ชื่อการทดสอบ', '')}"
        ),
    )

    selected_document = (
        search_documents[
            selected_index
        ]
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
                "-"
            )
        )

    with col2:

        st.write(
            "### ชื่อการทดสอบ"
        )

        st.write(
            selected_document.get(
                "ชื่อการทดสอบ",
                "-"
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
                ""
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
                []
            )
        )

        if vector:

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:

                st.metric(
                    "Dimensions",
                    len(vector)
                )

            with col2:

                st.metric(
                    "Model",
                    selected_document.get(
                        MODEL_FIELD,
                        "-"
                    )
                )

            with col3:

                st.metric(
                    "Vector Size",
                    f"{len(vector)} numbers"
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
# 23. Preview Data
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
        1
    )
    .limit(10)
)


if preview_documents:

    preview_rows = []

    for item in preview_documents:

        vector = item.get(
            VECTOR_FIELD,
            []
        )

        preview_rows.append(
            {
                "รหัส": item.get(
                    "รหัส"
                ),

                "ชื่อการทดสอบ": item.get(
                    "ชื่อการทดสอบ"
                ),

                "ข้อบ่งชี้การทดสอบ": item.get(
                    "ข้อบ่งชี้การทดสอบ"
                ),

                "มี Vector": (
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
# 24. Preview One Embedding Text
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
# 25. Vectorize Options
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
# 26. Count Target Documents
# ============================================================

target_count = (
    collection.count_documents(
        vectorize_query
    )
)

st.info(
    f"ข้อมูลที่จะประมวลผล: "
    f"**{target_count:,} รายการ**"
)


# ============================================================
# 27. Generate Vector Button
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
            text="กำลังเตรียมข้อมูล..."
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
                1
            )
        )

        # ----------------------------------------------------
        # Process documents
        # ----------------------------------------------------

        for index, document in enumerate(
            documents,
            start=1,
        ):

            try:

                test_code = document.get(
                    "รหัส",
                    ""
                )

                test_name = document.get(
                    "ชื่อการทดสอบ",
                    ""
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
                # Create text
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
                # Convert numpy array -> list
                # ------------------------------------------------

                vector_list = (
                    vector.tolist()
                )

                # ------------------------------------------------
                # Update MongoDB
                # ------------------------------------------------

                collection.update_one(
                    {
                        "_id": document["_id"]
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
        # Show errors
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
# 28. Show Vector Data
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
        1
    )
    .limit(10)
)


if vector_sample:

    vector_preview_rows = []

    for item in vector_sample:

        vector = item.get(
            VECTOR_FIELD,
            []
        )

        vector_preview_rows.append(
            {
                "รหัส": item.get(
                    "รหัส"
                ),

                "ชื่อการทดสอบ": item.get(
                    "ชื่อการทดสอบ"
                ),

                "Model": item.get(
                    MODEL_FIELD
                ),

                "Dimensions": item.get(
                    DIMENSION_FIELD
                ),

                "Vector Preview": (
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
# 29. Refresh
# ============================================================

st.divider()

if st.button(
    "🔄 Refresh",
    width="stretch",
):

    st.rerun()


# ============================================================
# 30. Footer
# ============================================================

st.divider()

st.caption(
    "Lab Test Vectorizer • "
    "MongoDB Atlas • "
    "Sentence Transformers"
)