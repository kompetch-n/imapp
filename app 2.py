# ============================================================
# Streamlit - MongoDB Lab Test Vectorizer
#
# Database  : lab_db
# Collection: test_catalog
#
# Function:
# 1. Connect MongoDB Atlas
# 2. Read lab test data
# 3. Create embedding text
# 4. Convert text -> vector
# 5. Save vector back to MongoDB
# 6. Preview vector data
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

# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
# Do NOT put your real MongoDB password directly in source code.
#
# Recommended:
#
# MONGODB_URI = st.secrets["MONGODB_URI"]
#
# Create:
#
# .streamlit/secrets.toml
#
# MONGODB_URI = "mongodb+srv://..."
#
# ------------------------------------------------------------

MONGODB_URI = st.secrets.get(
    "MONGODB_URI",
    ""
)

DATABASE_NAME = "lab_db"
COLLECTION_NAME = "test_catalog"


# ============================================================
# 3. Embedding Model
# ============================================================

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

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
# 5. Header
# ============================================================

st.title("🧬 Lab Test Vectorizer")

st.markdown(
    """
    เครื่องมือสำหรับแปลงข้อมูลการตรวจทางห้องปฏิบัติการ
    จาก MongoDB Atlas ให้เป็น **Vector Embedding**

    Vector ที่สร้างจะถูกบันทึกกลับเข้า collection
    `test_catalog`
    """
)


# ============================================================
# 6. Sidebar
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.write("Database")
    st.code(DATABASE_NAME)

    st.write("Collection")
    st.code(COLLECTION_NAME)

    st.write("Embedding Model")
    st.code(MODEL_NAME)

    st.write("Vector Dimensions")
    st.info("384 dimensions")

    st.divider()

    st.subheader("Fields used for Vector")

    for field in EMBEDDING_FIELDS:
        st.write(f"• {field}")


# ============================================================
# 7. Check MongoDB URI
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
# 8. Connect MongoDB
# ============================================================

@st.cache_resource
def get_mongodb_collection():

    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=10000,
    )

    # Test connection
    client.admin.command("ping")

    db = client[DATABASE_NAME]

    collection = db[COLLECTION_NAME]

    return client, collection


# ============================================================
# 9. Load Embedding Model
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        MODEL_NAME
    )

    return model


# ============================================================
# 10. Connect
# ============================================================

try:

    client, collection = get_mongodb_collection()

    st.success(
        "🟢 MongoDB Atlas connection: OK"
    )

except Exception as e:

    st.error(
        "ไม่สามารถเชื่อมต่อ MongoDB Atlas ได้"
    )

    st.exception(e)

    st.stop()


# ============================================================
# 11. Load Model
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

    st.exception(e)

    st.stop()


# ============================================================
# 12. Model Information
# ============================================================

try:

    vector_dimension = model.get_sentence_embedding_dimension()

except Exception:

    vector_dimension = 384


# ============================================================
# 13. Get MongoDB Statistics
# ============================================================

total_documents = collection.count_documents({})

vector_documents = collection.count_documents(
    {
        VECTOR_FIELD: {
            "$exists": True
        }
    }
)

remaining_documents = total_documents - vector_documents


# ============================================================
# 14. Statistics
# ============================================================

st.subheader("📊 Database Status")

col1, col2, col3, col4 = st.columns(4)

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
# 15. Create Embedding Text
# ============================================================

def safe_text(value):

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def create_embedding_text(document):

    parts = []

    for field in EMBEDDING_FIELDS:

        value = safe_text(
            document.get(field)
        )

        if value:

            parts.append(
                f"{field}: {value}"
            )

    return "\n".join(parts)


# ============================================================
# 16. Preview Data
# ============================================================

st.divider()

st.subheader(
    "👀 Preview Embedding Text"
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

    preview_df = pd.DataFrame(
        preview_documents
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
# 17. Preview One Embedding Text
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

    sample_text = create_embedding_text(
        sample_document
    )

    st.text_area(
        "Embedding Text",
        value=sample_text,
        height=250,
    )


# ============================================================
# 18. Vectorize Options
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


if option == "เฉพาะข้อมูลที่ยังไม่มี Vector":

    query = {
        VECTOR_FIELD: {
            "$exists": False
        }
    }

else:

    query = {}


# ============================================================
# 19. Count Target Documents
# ============================================================

target_count = collection.count_documents(
    query
)

st.info(
    f"ข้อมูลที่จะประมวลผล: **{target_count:,} รายการ**"
)


# ============================================================
# 20. Generate Vector Button
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
                query,
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
                    f"กำลังประมวลผล "
                    f"**{index:,} / {target_count:,}**  "
                    f"รหัส: `{test_code}`  "
                    f"{test_name}"
                )

                # ------------------------------------------------
                # Create text
                # ------------------------------------------------

                embedding_text = create_embedding_text(
                    document
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
                # Convert numpy array -> Python list
                # ------------------------------------------------

                vector_list = vector.tolist()

                # ------------------------------------------------
                # Update MongoDB
                # ------------------------------------------------

                collection.update_one(
                    {
                        "_id": document["_id"]
                    },
                    {
                        "$set": {
                            TEXT_FIELD: embedding_text,

                            VECTOR_FIELD: vector_list,

                            MODEL_FIELD: MODEL_NAME,

                            DIMENSION_FIELD: vector_dimension,
                        }
                    },
                )

                success_count += 1

            except Exception as e:

                error_count += 1

                errors.append(
                    {
                        "รหัส": document.get(
                            "รหัส"
                        ),

                        "ชื่อการทดสอบ": document.get(
                            "ชื่อการทดสอบ"
                        ),

                        "error": str(e),
                    }
                )

            # ------------------------------------------------
            # Update progress
            # ------------------------------------------------

            percent = index / target_count

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

        # ----------------------------------------------------
        # Rerun
        # ----------------------------------------------------

        st.rerun()


# ============================================================
# 21. Show Vector Data
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

    vector_df = pd.DataFrame(
        vector_sample
    )

    # --------------------------------------------------------
    # Vector preview
    # --------------------------------------------------------

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

                "Vector Preview": str(
                    vector[:10]
                ) + " ...",
            }
        )

    vector_preview_df = pd.DataFrame(
        vector_preview_rows
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
# 22. Select Test for Vector Detail
# ============================================================

st.subheader(
    "🔍 ดู Vector ของรายการ"
)

test_codes = list(
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
        },
    )
    .sort(
        "รหัส",
        1
    )
)


if test_codes:

    code_options = [
        item.get("รหัส")
        for item in test_codes
    ]

    selected_code = st.selectbox(
        "เลือกรหัสการทดสอบ",
        code_options,
    )

    selected_document = collection.find_one(
        {
            "รหัส": selected_code,
            VECTOR_FIELD: {
                "$exists": True
            },
        },
        {
            "_id": 0
        },
    )

    if selected_document:

        st.write(
            "###",
            selected_document.get(
                "ชื่อการทดสอบ",
                "-"
            ),
        )

        st.write(
            "Embedding Text"
        )

        st.text_area(
            "text",
            value=selected_document.get(
                TEXT_FIELD,
                ""
            ),
            height=200,
            label_visibility="collapsed",
        )

        vector = selected_document.get(
            VECTOR_FIELD,
            []
        )

        col1, col2, col3 = st.columns(3)

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


# ============================================================
# 23. Refresh Button
# ============================================================

st.divider()

if st.button(
    "🔄 Refresh",
    width="stretch",
):

    st.rerun()


# ============================================================
# 24. Footer
# ============================================================

st.divider()

st.caption(
    "Lab Test Vectorizer • MongoDB Atlas • "
    "Sentence Transformers"
)