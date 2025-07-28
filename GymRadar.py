# Import thư viện Gemini AI
import google.generativeai as genai

# Import thư viện làm việc với hệ điều hành
import os

# Import thư viện để tải biến môi trường
from dotenv import load_dotenv

# Import FastAPI
from fastapi import FastAPI

# Import thư viện kết nối với SQL Server
import pyodbc

# Import Pydantic
from pydantic import BaseModel

# Thêm middleware CORS
from fastapi.middleware.cors import CORSMiddleware

# Tải biến môi trường
load_dotenv()

# Cấu hình API key cho Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Chuỗi kết nối đến SQL Server
conn_str = (
    r"DRIVER={ODBC Driver 17 for SQL Server};"
    f"SERVER={os.getenv('DB_SERVER')};"
    f"DATABASE={os.getenv('DB_NAME')};"
    f"UID={os.getenv('DB_USER')};"
    f"PWD={os.getenv('DB_PASSWORD')};"
)

# Kết nối database
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Khởi tạo mô hình Gemini
model = genai.GenerativeModel("gemini-2.0-flash")

# Định nghĩa schema dữ liệu
class ChatRequest(BaseModel):
    prompt: str
    longitude: float | None = None
    latitude: float | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Tìm phòng gym ở gần",
                "longitude": 106.700981,
                "latitude": 10.776889
            }
        }

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="GymRadar Chatbot API",
    description="API chatbot sử dụng Gemini và SQL Server để gợi ý phòng gym.",
    version="1.0.4"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hàm truy vấn cơ sở dữ liệu
def query_database(query):
    try:
        print(f"Executing query: {query}")
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"Query results: {results}")
        return results
    except pyodbc.Error as e:
        print(f"Database error: {str(e)}")
        return f"Lỗi cơ sở dữ liệu: {str(e)}"

# Tạo query SQL tìm gym gần vị trí người dùng
def build_nearby_gym_query(longitude, latitude, max_distance_km=5):
    return f"""
    SELECT *
FROM (
    SELECT 
        Id, GymName, Since, Address, RepresentName, TaxCode, 
        CAST(Longitude AS FLOAT) AS Longitude,
        CAST(Latitude AS FLOAT) AS Latitude,
        QRCode, HotResearch, AccountId, Active, 
        CreateAt, UpdateAt, DeleteAt, MainImage,
        6371 * acos(
            cos(radians(10.805765)) *
            cos(radians(CAST(Latitude AS FLOAT))) *
            cos(radians(CAST(Longitude AS FLOAT)) - radians(106.741796)) +
            sin(radians(10.805765)) *
            sin(radians(CAST(Latitude AS FLOAT)))
        ) AS distance_km
    FROM dbo.Gym
    WHERE Active = 1
) AS gyms
WHERE gyms.distance_km <= 5
ORDER BY gyms.distance_km ASC;

    """

# Hàm phân loại prompt có cần query DB hay không
def classify_query(user_input):
    prompt = f"""
    Xác định xem câu hỏi có yêu cầu truy vấn cơ sở dữ liệu không.
    Nếu có, trả về câu lệnh SELECT phù hợp. Nếu không, trả về 'NO_DB_QUERY'.

    Dữ liệu nằm ở bảng dbo.Gym với các cột: Id, GymName, Since, Address, RepresentName, TaxCode,
    Longitude, Latitude, QRCode, HotResearch, AccountId, Active, CreateAt, UpdateAt, DeleteAt, MainImage.

    Câu hỏi: {user_input}

    Chỉ trả về SQL hoặc 'NO_DB_QUERY'.
    """

    try:
        response = model.generate_content(prompt)
        lines = [line.strip() for line in response.text.splitlines() if line.strip()]
        result = " ".join(lines).replace("```sql", "").replace("```", "").strip()

        if not result or result == "NO_DB_QUERY":
            return False, None
        if result.lower().startswith("select") and "from dbo.gym" in result.lower():
            return True, result

        return False, None
    except Exception as e:
        print(f"Gemini error: {str(e)}")
        return False, None

# Hàm xử lý chính
def get_response(user_input, longitude=None, latitude=None):
    try:
        # Xử lý nếu câu hỏi yêu cầu tìm gần và có tọa độ
        if longitude and latitude and "gần" in user_input.lower():
            sql_query = build_nearby_gym_query(longitude, latitude)
            results = query_database(sql_query)

            if isinstance(results, str) or not results:
                return {"promptResponse": "GymRadar xin lỗi, không tìm thấy phòng gym nào gần bạn."}

            gyms = []
            for row in results:
                gyms.append({
                    "id": str(row.Id),
                    "gymName": row.GymName,
                    "since": row.Since,
                    "address": row.Address,
                    "representName": row.RepresentName,
                    "taxCode": row.TaxCode,
                    "longitude": row.Longitude,
                    "latitude": row.Latitude,
                    "qrCode": row.QRCode,
                    "hotResearch": row.HotResearch,
                    "accountId": str(row.AccountId),
                    "active": row.Active,
                    "createAt": row.CreateAt.isoformat() if row.CreateAt else None,
                    "updateAt": row.UpdateAt.isoformat() if row.UpdateAt else None,
                    "deleteAt": row.DeleteAt.isoformat() if row.DeleteAt else None,
                    "mainImage": row.MainImage,
                    "distance_km": round(row.distance_km, 2)
                })

            prompt_response = "GymRadar gợi ý các phòng gym gần bạn:\n" + "\n".join(
                [f"{g['gymName']} ({g['distance_km']} km)" for g in gyms]
            )
            return {"gyms": gyms, "promptResponse": prompt_response}

        # Nếu không, phân loại bình thường
        is_db_query, sql_query = classify_query(user_input)
        if is_db_query:
            results = query_database(sql_query)
            if isinstance(results, str) or not results:
                return {"promptResponse": "GymRadar xin lỗi, không có phòng gym phù hợp với yêu cầu của bạn."}

            gyms = []
            for row in results:
                gyms.append({
                    "id": str(row.Id),
                    "gymName": row.GymName,
                    "since": row.Since,
                    "address": row.Address,
                    "representName": row.RepresentName,
                    "taxCode": row.TaxCode,
                    "longitude": row.Longitude,
                    "latitude": row.Latitude,
                    "qrCode": row.QRCode,
                    "hotResearch": row.HotResearch,
                    "accountId": str(row.AccountId),
                    "active": row.Active,
                    "createAt": row.CreateAt.isoformat() if row.CreateAt else None,
                    "updateAt": row.UpdateAt.isoformat() if row.UpdateAt else None,
                    "deleteAt": row.DeleteAt.isoformat() if row.DeleteAt else None,
                    "mainImage": row.MainImage
                })

            if len(gyms) == 1:
                g = gyms[0]
                prompt_response = f"GymRadar gợi ý bạn ghé thăm {g['gymName']} tại {g['address']}."
            else:
                names = ", ".join([g["gymName"] for g in gyms])
                prompt_response = f"GymRadar gợi ý các phòng gym sau: {names}."
            return {"gyms": gyms, "promptResponse": prompt_response}

        # Trả lời tự do bằng Gemini
        project_context = "Bạn là chatbot GymRadar, trả lời thân thiện, tự nhiên, bằng tiếng Việt."
        prompt = f"{project_context}\nCâu hỏi: {user_input}"
        response = model.generate_content(prompt)
        return {"promptResponse": response.text}

    except Exception as e:
        print(f"Lỗi trong get_response: {str(e)}")
        return {"promptResponse": "GymRadar xin lỗi, đã xảy ra lỗi."}

# API endpoint
@app.post("/chat", summary="Gửi câu hỏi đến chatbot", response_description="Trả về câu trả lời")
async def chat(request: ChatRequest):
    return get_response(request.prompt, request.longitude, request.latitude)

# Chạy ứng dụng
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
