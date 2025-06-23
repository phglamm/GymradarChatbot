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
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "gợi ý phòng gym ở địa chỉ Ways Station"
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
        print(f"Executing query: {query}")  # Debug the exact query
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"Query results: {results}")  # Debug results
        return results
    except pyodbc.Error as e:
        print(f"Database error details: {str(e)}")  # Debug full error
        return f"Lỗi cơ sở dữ liệu: {str(e)}"

# Hàm phân loại câu hỏi
def classify_query(user_input):
    prompt = f"""
    Xác định xem câu hỏi có yêu cầu truy vấn cơ sở dữ liệu không.
    Cơ sở dữ liệu có một bảng:
    - 'dbo.Gym' với cột: Id, GymName, Since, Address, RepresentName, TaxCode, Longitude, Latitude, QRCode, HotResearch, AccountId, Active, CreateAt, UpdateAt, DeleteAt, MainImage.
    - Nếu câu hỏi yêu cầu thông tin từ Gym (ví dụ: gợi ý phòng gym theo địa chỉ như 'Ways Station', tên phòng gym như 'Gym Next', hoặc trạng thái 'Active'), trả về MỘT câu lệnh SELECT theo cú pháp SQL Server.
      - Luôn SELECT các cột từ Gym: Id, GymName, Since, Address, RepresentName, TaxCode, Longitude, Latitude, QRCode, HotResearch, AccountId, Active, CreateAt, UpdateAt, DeleteAt, MainImage.
      - Sử dụng WHERE để lọc theo Address, GymName, hoặc Active (nếu đề cập đến trạng thái).
      - Nếu lọc theo Address hoặc GymName, sử dụng LIKE '%value%' để cho phép biến thể (ví dụ: LIKE '%Ways Station%').
      - Nếu nhiều tiêu chí (ví dụ: 'Phòng gym ở Ways Station và Active'), kết hợp với AND.
      - Nếu không đủ tiêu chí rõ ràng (ví dụ: 'Gợi ý phòng gym'), thử lọc theo GymName LIKE '%Gym%'.
      - Nếu không tìm thấy kết quả, trả về truy vấn tối thiểu dựa trên tiêu chí rõ ràng nhất (ví dụ: chỉ Address).
      - Ví dụ:
        - 'Gợi ý phòng gym ở Ways Station' trả về:
          SELECT Id, GymName, Since, Address, RepresentName, TaxCode, Longitude, Latitude, QRCode, HotResearch, AccountId, Active, CreateAt, UpdateAt, DeleteAt, MainImage 
          FROM dbo.Gym 
          WHERE Address LIKE '%Ways Station%'.
        - 'Gợi ý phòng gym tên Gym Next' trả về:
          SELECT Id, GymName, Since, Address, RepresentName, TaxCode, Longitude, Latitude, QRCode, HotResearch, AccountId, Active, CreateAt, UpdateAt, DeleteAt, MainImage 
          FROM dbo.Gym 
          WHERE GymName LIKE '%Gym Next%'.
        - 'Gợi ý phòng gym Active' trả về:
          SELECT Id, GymName, Since, Address, RepresentName, TaxCode, Longitude, Latitude, QRCode, HotResearch, AccountId, Active, CreateAt, UpdateAt, DeleteAt, MainImage 
          FROM dbo.Gym 
          WHERE Active = 1.
    - Nếu không, trả về 'NO_DB_QUERY'.
    Câu hỏi: {user_input}

    Chỉ trả về SQL hoặc 'NO_DB_QUERY'.
    """
    try:
        response = model.generate_content(prompt)
        print(f"Raw Gemini response: {response.text}")  # Debug raw output
        lines = [line.strip() for line in response.text.splitlines() if line.strip()]
        result = " ".join(lines).replace("```sql", "").replace("```", "").strip()
        print(f"Processed query: {result}")  # Debug processed output

        if not result:
            print("Empty query after processing")
            return False, None

        if result == "NO_DB_QUERY":
            return False, None

        if result.lower().startswith("select"):
            # Kiểm tra cú pháp SQL cơ bản
            if "from dbo.gym" in result.lower():
                return True, result
            else:
                print(f"Invalid SQL: Missing required table, full text: {result}")
                return False, None
        else:
            print(f"Validation failed: Query does not start with 'select', full text: {result}")
            return False, None

    except Exception as e:
        print(f"Gemini error: {str(e)}")
        return False, None

# Hàm xử lý và trả về câu trả lời
def get_response(user_input):
    try:
        is_db_query, sql_query = classify_query(user_input)

        project_context = """
        Bạn là chatbot của GymRadar. Luôn sử dụng 'GymRadar' làm chủ ngữ. 
        Bạn là chatbot của ứng dụng GymRadar là ứng dụng cho phép người dùng tìm kiếm, đăng ký phòng Gym, các khoá tập,...
        Bạn sẽ hiểu rõ hơn về vấn đề sức khoẻ, Gym.
        Trả lời thân thiện, tự nhiên bằng tiếng Việt.
        """

        if is_db_query:
            print("Truy vấn SQL:", sql_query)
            results = query_database(sql_query)

            if isinstance(results, str) or not results:
                return {"promptResponse": "GymRadar xin lỗi, hiện tại không có phòng gym phù hợp với mô tả của bạn."}

            if len(results) == 1:
                row = results[0]
                gym = {
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
                }
                prompt_response = f"GymRadar gợi ý bạn ghé thăm {row.GymName} tại {row.Address}. Phòng gym này hoạt động từ {row.Since}!"
                return {
                    "gym": gym,
                    "promptResponse": prompt_response
                }
            else:
                gyms = [
                    {
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
                    } for row in results
                ]
                gym_names = ", ".join([row.GymName for row in results])
                prompt_response = f"GymRadar gợi ý bạn ghé thăm: {gym_names}. Các phòng gym này đều có địa chỉ đáng chú ý!"
                return {
                    "gyms": gyms,
                    "promptResponse": prompt_response
                }

        else:
            prompt = f"""
            {project_context}
            Trả lời câu hỏi sau tự nhiên, bằng tiếng Việt:
            Câu hỏi: {user_input}
            """
            final_response = model.generate_content(prompt)
            return {"promptResponse": final_response.text}

    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        return {"promptResponse": "GymRadar xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại!"}

# Endpoint POST /chat
@app.post("/chat", summary="Gửi câu hỏi đến chatbot", response_description="Trả về câu trả lời")
async def chat(request: ChatRequest):
    response = get_response(request.prompt)
    return response

# Chạy ứng dụng
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)