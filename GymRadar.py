# Import các thư viện cốt lõi
import json
import os
import math
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import List, Optional

# Import các thư viện bên ngoài
import google.generativeai as genai
import pyodbc
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Tải biến môi trường
load_dotenv()

# Cấu hình Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Cấu hình cơ sở dữ liệu
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')

# Xây dựng chuỗi kết nối
if not db_user and not db_password:
    conn_str = (
        r"DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={os.getenv('DB_SERVER')};"
        f"DATABASE={os.getenv('DB_NAME')};"
        "Trusted_Connection=yes;"
    )
else:
    conn_str = (
        r"DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={os.getenv('DB_SERVER')};"
        f"DATABASE={os.getenv('DB_NAME')};"
        f"UID={db_user};"
        f"PWD={db_password};"
    )

# Kết nối cơ sở dữ liệu
try:
    print("Đang kết nối đến cơ sở dữ liệu...")
    print(f"Server: {os.getenv('DB_SERVER')}")
    print(f"Database: {os.getenv('DB_NAME')}")
    
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    print("Kết nối cơ sở dữ liệu thành công!")
    
except pyodbc.Error as e:
    print(f"Kết nối cơ sở dữ liệu thất bại: {e}")
    print("Vui lòng kiểm tra cấu hình cơ sở dữ liệu")
    conn = None
    cursor = None

# Khởi tạo mô hình Gemini
model = genai.GenerativeModel("gemini-2.0-flash")

class ChatRequest(BaseModel):
    prompt: str
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    conversation_history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    promptResponse: str
    gyms: Optional[List[dict]] = None
    conversation_history: Optional[List[dict]] = []

def build_conversation_context(conversation_history: List[dict]) -> str:
    """Xây dựng ngữ cảnh từ lịch sử hội thoại"""
    if not conversation_history:
        return ""
    
    context_parts = []
    for msg in conversation_history[-10:]:  # Lấy 10 tin nhắn gần nhất làm ngữ cảnh
        role_label = "Người dùng" if msg.get("role") == "user" else "GymRadar"
        content = sanitize_text_for_json(msg.get('content', ''))
        context_parts.append(f"{role_label}: {content}")
    
    return "\n".join(context_parts)

def sanitize_text_for_json(text: str) -> str:
    """Làm sạch text để tránh lỗi JSON parsing"""
    if not text:
        return ""
    
    # Thay thế các ký tự điều khiển có thể gây lỗi JSON
    try:
        # Loại bỏ các ký tự điều khiển không mong muốn
        cleaned_text = text.replace('\x00', '').replace('\x08', '').replace('\x0c', '')
        cleaned_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned_text)
        
        # Đảm bảo text có thể được encode thành JSON
        json.dumps(cleaned_text)
        return cleaned_text
    except (UnicodeDecodeError, json.JSONDecodeError):
        # Nếu vẫn có lỗi, encode/decode để loại bỏ ký tự không hợp lệ
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')

def safe_get_row_data(row):
    """Trích xuất dữ liệu từ hàng cơ sở dữ liệu một cách an toàn"""
    try:
        def get_attr(attr_name, default=None):
            try:
                return getattr(row, attr_name, default)
            except AttributeError:
                return default
        
        def get_bool_attr(attr_name, default=False):
            try:
                value = getattr(row, attr_name, default)
                if value is None:
                    return default
                if isinstance(value, int):
                    return bool(value)
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value)
            except AttributeError:
                return default
        
        def format_date(date_field):
            if date_field:
                try:
                    return date_field.isoformat() if hasattr(date_field, 'isoformat') else str(date_field)
                except:
                    return None
            return None
        
        gym_data = {
            "id": str(get_attr('Id', '')),
            "gymName": get_attr('GymName', ''),
            "since": format_date(get_attr('Since')),
            "address": get_attr('Address', ''),
            "representName": get_attr('RepresentName', ''),
            "taxCode": get_attr('TaxCode', ''),
            "longitude": get_attr('Longitude'),
            "latitude": get_attr('Latitude'),
            "qrCode": get_attr('QRCode', ''),
            "hotResearch": get_bool_attr('HotResearch', False),
            "accountId": str(get_attr('AccountId', '')) if get_attr('AccountId') else None,
            "active": get_bool_attr('Active', True),
            "createAt": format_date(get_attr('CreateAt')),
            "updateAt": format_date(get_attr('UpdateAt')),
            "deleteAt": format_date(get_attr('DeleteAt')),
            "mainImage": get_attr('MainImage', '')
        }
        
        # Handle distance for nearby queries
        distance = get_attr('distance_km')
        if distance is not None:
            try:
                gym_data["distance_km"] = round(float(distance), 2)
            except:
                gym_data["distance_km"] = 0
        
        return gym_data
        
    except Exception as e:
        print(f"Lỗi trong safe_get_row_data: {str(e)}")
        return {
            "id": "",
            "gymName": "Phòng gym không xác định",
            "since": None,
            "address": "",
            "representName": "",
            "taxCode": "",
            "longitude": None,
            "latitude": None,
            "qrCode": "",
            "hotResearch": False,
            "accountId": None,
            "active": True,
            "createAt": None,
            "updateAt": None,
            "deleteAt": None,
            "mainImage": ""
        }

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="GymRadar Chatbot AI",
    description="Chatbot tìm kiếm phòng gym thông minh sử dụng Gemini AI và SQL Server",
    version="2.0.0"
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
    if conn is None or cursor is None:
        print("Kết nối cơ sở dữ liệu không khả dụng")
        return "Lỗi kết nối cơ sở dữ liệu"
    
    try:
        print(f"Đang thực thi truy vấn: {query}")
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"Kết quả truy vấn: {len(results)} hàng")
        return results
    except pyodbc.Error as e:
        print(f"Lỗi cơ sở dữ liệu: {str(e)}")
        return f"Lỗi cơ sở dữ liệu: {str(e)}"

# Xây dựng truy vấn SQL để tìm gym gần
def build_nearby_gym_query(longitude, latitude, max_distance_km= 10):
    """Xây dựng truy vấn SQL để tìm gym gần sử dụng công thức Haversine"""
    # Tối ưu hóa với bounding box để giảm tải tính toán
    lat_range = max_distance_km / 111.0  # 1 độ vĩ độ ≈ 111km
    lng_range = max_distance_km / (111.0 * abs(math.cos(math.radians(latitude))))
    
    return f"""
    WITH BoundedGyms AS (
        SELECT 
            Id, GymName, Since, Address, RepresentName, TaxCode,
            CAST(Longitude AS FLOAT) AS Longitude,
            CAST(Latitude AS FLOAT) AS Latitude,
            QRCode, HotResearch, AccountId, Active,
            CreateAt, UpdateAt, DeleteAt, MainImage
        FROM dbo.Gym 
        WHERE Active = 1
        AND CAST(Latitude AS FLOAT) BETWEEN {latitude - lat_range} AND {latitude + lat_range}
        AND CAST(Longitude AS FLOAT) BETWEEN {longitude - lng_range} AND {longitude + lng_range}
    ),
    DistanceCalculated AS (
        SELECT *,
            6371.0 * 2 * ASIN(
                SQRT(
                    POWER(SIN(RADIANS({latitude} - Latitude) / 2), 2) +
                    COS(RADIANS({latitude})) * COS(RADIANS(Latitude)) *
                    POWER(SIN(RADIANS({longitude} - Longitude) / 2), 2)
                )
            ) AS distance_km
        FROM BoundedGyms
    )
    SELECT * 
    FROM DistanceCalculated
    WHERE distance_km <= {max_distance_km}
    ORDER BY distance_km ASC, HotResearch DESC, GymName ASC;
    """

def get_nearby_distance_preference(user_input):
    """Phân tích đầu vào của người dùng để xác định bán kính tìm kiếm phù hợp"""
    user_input_lower = user_input.lower()
    
    # 1. Tìm số km cụ thể trong câu (ưu tiên cao nhất)
    import re
    km_match = re.search(r'(\d+)\s*km', user_input_lower)
    if km_match:
        distance = int(km_match.group(1))
        # Giới hạn khoảng cách hợp lý (1-50km)
        return max(1, min(distance, 50))
    
    # 2. Phân tích theo cấp độ khoảng cách với từ khóa thông minh
    distance_patterns = {
        # Rất gần - 2km
        2: [
            r'(rất gần|very close|walking distance|đi bộ|đi bộ được)',
            r'(ngay gần|sát bên|cực gần|siêu gần)',
            r'(trong phạm vi \d{1,3}\s*m|dưới 1km|under 1km)'
        ],
        
        # Gần - 5km  
        5: [
            r'(gần|nearby|close|lân cận|kề bên)',
            r'(quanh đây|xung quanh|around here)',
            r'(không xa|not far|gần nhà|near home)'
        ],
        
        # Trung bình - 10km
        10: [
            r'(khu vực|trong khu|in the area|local)',
            r'(xa một chút|bit farther|hơi xa)',
            r'(trong thành phố|in the city|cùng thành phố)'
        ],
        
        # Xa - 15km
        15: [
            r'(xa hơn|farther|more distant)',
            r'(trong tỉnh|in province|cùng tỉnh)',
            r'(mở rộng|expand|extend)'
        ],
        
        # Rất xa - 25km
        25: [
            r'(rất xa|very far|distant)',
            r'(khắp nơi|everywhere|anywhere)',
            r'(toàn bộ|all|entire|whole)'
        ],
        
        # Không giới hạn - 50km
        50: [
            r'(tất cả|all gyms|mọi|every|bất kỳ đâu)',
            r'(không giới hạn|unlimited|no limit)',
            r'(toàn quốc|nationwide|whole country)'
        ]
    }
    
    # 3. Duyệt qua các pattern theo thứ tự ưu tiên
    for distance, patterns in distance_patterns.items():
        for pattern in patterns:
            if re.search(pattern, user_input_lower):
                return distance
    
    # 4. Phân tích thông minh dựa trên ngữ cảnh
    
    # Nếu có từ khóa về phương tiện di chuyển
    transport_keywords = {
        r'(xe đạp|bicycle|bike)': 8,
        r'(xe máy|motorbike|scooter)': 15, 
        r'(ô tô|car|drive|driving)': 20,
        r'(xe bus|bus|public transport)': 12
    }
    
    for pattern, distance in transport_keywords.items():
        if re.search(pattern, user_input_lower):
            return distance
    
    # Nếu có từ khóa về thời gian
    time_keywords = {
        r'(5 phút|5min|năm phút)': 3,
        r'(10 phút|10min|mười phút)': 5,
        r'(15 phút|15min|mười lăm phút)': 8,
        r'(20 phút|20min|hai mươi phút)': 12,
        r'(30 phút|30min|nửa giờ|half hour)': 18
    }
    
    for pattern, distance in time_keywords.items():
        if re.search(pattern, user_input_lower):
            return distance
    
    # 5. Phân tích theo địa danh cụ thể
    location_keywords = {
        r'(quận \d+|district \d+)': 8,        # Trong quận
        r'(thành phố|city|tp\.)': 15,         # Trong thành phố  
        r'(tỉnh|province|tỉnh thành)': 25,    # Trong tỉnh
        r'(huyện|county|suburban)': 20        # Ngoại thành
    }
    
    for pattern, distance in location_keywords.items():
        if re.search(pattern, user_input_lower):
            return distance
    
    # 6. Mặc định thông minh dựa trên độ dài câu
    if len(user_input_lower.split()) <= 3:
        return 8   # Câu ngắn -> tìm gần
    elif len(user_input_lower.split()) <= 6:
        return 10  # Câu trung bình -> tìm vừa
    else:
        return 12  # Câu dài -> có thể muốn tìm rộng hơn

def format_distance_friendly(distance_km):
    """Định dạng khoảng cách theo cách thân thiện với người dùng"""
    if distance_km < 0.5:
        return f"{int(distance_km * 1000)}m"
    elif distance_km < 1:
        return f"{distance_km:.1f}km (đi bộ được)" 
    elif distance_km < 3:
        return f"{distance_km:.1f}km (gần)"
    elif distance_km < 5:
        return f"{distance_km:.1f}km (xe đạp/xe máy)"
    elif distance_km < 10:
        return f"{distance_km:.1f}km (ô tô/xe máy)"
    else:
        return f"{distance_km:.1f}km (xa)"

def normalize_vietnamese_text(text):
    """Chuẩn hóa văn bản tiếng Việt để tìm kiếm tốt hơn"""
    if not text:
        return ""
    
    text = text.lower()
    
    # Bản đồ chuyển đổi ký tự tiếng Việt sang Latin
    char_map = {
        'à': 'a', 'á': 'a', 'ạ': 'a', 'ả': 'a', 'ã': 'a',
        'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ậ': 'a', 'ẩ': 'a', 'ẫ': 'a',
        'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ặ': 'a', 'ẳ': 'a', 'ẵ': 'a',
        'è': 'e', 'é': 'e', 'ẹ': 'e', 'ẻ': 'e', 'ẽ': 'e',
        'ê': 'e', 'ề': 'e', 'ế': 'e', 'ệ': 'e', 'ể': 'e', 'ễ': 'e',
        'ì': 'i', 'í': 'i', 'ị': 'i', 'ỉ': 'i', 'ĩ': 'i',
        'ò': 'o', 'ó': 'o', 'ọ': 'o', 'ỏ': 'o', 'õ': 'o',
        'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ộ': 'o', 'ổ': 'o', 'ỗ': 'o',
        'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ợ': 'o', 'ở': 'o', 'ỡ': 'o',
        'ù': 'u', 'ú': 'u', 'ụ': 'u', 'ủ': 'u', 'ũ': 'u',
        'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ự': 'u', 'ử': 'u', 'ữ': 'u',
        'ỳ': 'y', 'ý': 'y', 'ỵ': 'y', 'ỷ': 'y', 'ỹ': 'y',
        'đ': 'd'
    }
    
    # Thay thế ký tự tiếng Việt
    for viet_char, latin_char in char_map.items():
        text = text.replace(viet_char, latin_char)
    
    # Loại bỏ ký tự đặc biệt, chỉ giữ chữ cái và số
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = ' '.join(text.split())
    
    return text

def fuzzy_search_similarity(text1, text2):
    """Tính toán độ tương đồng giữa hai chuỗi văn bản"""
    if not text1 or not text2:
        return 0
    
    normalized1 = normalize_vietnamese_text(text1)
    normalized2 = normalize_vietnamese_text(text2)
    
    return SequenceMatcher(None, normalized1, normalized2).ratio()

def extract_search_keywords(user_input):
    """Trích xuất từ khóa tìm kiếm từ đầu vào của người dùng"""
    stop_words = {
        'tìm', 'tìm kiếm', 'find', 'search', 'có', 'không', 'gym', 'phòng gym', 
        'nào', 'đâu', 'where', 'what', 'gì', 'là', 'ở', 'tại', 'trong', 'của',
        'một', 'vài', 'những', 'các', 'the', 'a', 'an', 'and', 'or', 'for'
    }
    
    normalized = normalize_vietnamese_text(user_input)
    words = normalized.split()
    keywords = [word for word in words if word not in stop_words and len(word) >= 2]
    
    return keywords

def intelligent_gym_search(user_input):
    """Tìm kiếm gym thông minh với khả năng phân tích ngữ nghĩa nâng cao"""
    try:
        # Kiểm tra đầu vào
        if not user_input or not isinstance(user_input, str):
            return None
        
        user_input_lower = user_input.lower()
        
        # 1. Danh sách từ khóa chỉ rõ KHÔNG cần tìm kiếm gym (mở rộng)
        non_gym_indicators = [
            # Chào hỏi và lịch sự
            'xin chào', 'hello', 'hi', 'chào bạn', 'hey there',
            'cảm ơn', 'thank you', 'thanks', 'cám ơn', 'tks',
            'tạm biệt', 'bye', 'goodbye', 'see you',
            
            # Câu hỏi cá nhân  
            'tên tôi', 'my name', 'lặp lại tên', 'nhắc lại tên',
            'tôi là ai', 'who am i', 'remember me',
            
            # Tư vấn sức khỏe không cần gym cụ thể
            'làm sao để', 'how to', 'cách để', 'how can i',
            'ăn gì để', 'what to eat', 'what should i eat',
            'bài tập nào', 'what exercise', 'which workout',
            'tăng cân', 'giảm cân', 'lose weight', 'gain weight',
            'thời tiết', 'weather', 'nhiệt độ', 'temperature',
            
            # Phản hồi chung
            'ok', 'được rồi', 'tốt', 'good', 'fine', 'great',
            'đồng ý', 'agree', 'yes', 'no problem'
        ]
        
        # Nếu có từ khóa không liên quan gym, return None ngay
        if any(indicator in user_input_lower for indicator in non_gym_indicators):
            return None
        
        # 2. Phân tích ý định tìm kiếm gym (cải thiện)
        gym_search_patterns = [
            # Tìm kiếm trực tiếp
            r'(tìm|find|search|looking for)\s*(gym|phòng gym|fitness|thể dục)',
            r'(gym|phòng gym|fitness)\s*(nào|what|which|where)',
            r'(có|is there|are there)\s*(gym|phòng gym|fitness)',
            
            # Tìm kiếm theo địa điểm
            r'(gym|phòng gym|fitness)\s*(ở|at|in|near|gần)\s*(\w+)',
            r'(quận|district|huyện|thành phố|city)\s*\d*.*?(gym|phòng gym|fitness)',
            
            # Tìm kiếm theo đặc điểm
            r'(gym|phòng gym|fitness)\s*(hot|nổi tiếng|phổ biến|tốt|best)',
            r'(hot|nổi tiếng|phổ biến|tốt|best)\s*(gym|phòng gym|fitness)',
            
            # Tìm kiếm mở
            r'(danh sách|list)\s*(gym|phòng gym|fitness)',
            r'(tất cả|all)\s*(gym|phòng gym|fitness)',
            r'(những|the)\s*(gym|phòng gym|fitness)\s*(nào|what)'
        ]
        
        # Kiểm tra xem có khớp với pattern tìm kiếm gym không
        has_gym_search_intent = any(re.search(pattern, user_input_lower) for pattern in gym_search_patterns)
        
        # Nếu không có ý định tìm kiếm gym rõ ràng, return None
        if not has_gym_search_intent:
            return None
        
        # 3. Trích xuất thông tin tìm kiếm thông minh
        search_info = {
            'keywords': [],
            'location': None,
            'hot_search': False,
            'search_type': 'general'
        }
        
        # Trích xuất từ khóa quan trọng (bỏ stop words)
        stop_words = {
            'tìm', 'find', 'search', 'có', 'không', 'nào', 'đâu', 'where', 
            'what', 'gì', 'là', 'ở', 'tại', 'trong', 'của', 'một', 'vài', 
            'những', 'các', 'the', 'a', 'an', 'and', 'or', 'for', 'gym', 
            'phòng', 'fitness', 'center', 'club'
        }
        
        words = re.findall(r'\b\w+\b', normalize_vietnamese_text(user_input))
        keywords = [word for word in words if word not in stop_words and len(word) >= 2]
        search_info['keywords'] = keywords[:5]  # Lấy tối đa 5 từ khóa quan trọng nhất
        
        # Phát hiện tìm kiếm hot/phổ biến
        hot_patterns = [
            r'(hot|nổi tiếng|phổ biến|được yêu thích|tốt nhất|best|top)',
            r'(recommend|gợi ý|đề xuất|suggest)'
        ]
        search_info['hot_search'] = any(re.search(pattern, user_input_lower) for pattern in hot_patterns)
        
        # Phát hiện địa điểm cụ thể
        location_patterns = [
            r'(quận|district)\s*(\d+)',
            r'(huyện|county)\s*(\w+)',
            r'(thành phố|city|tp\.?)\s*(\w+)',
            r'(ở|tại|in|at)\s*(\w+)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                search_info['location'] = match.group(0)
                break
        
        # 4. Xây dựng truy vấn SQL thông minh
        base_conditions = ["Active = 1"]
        
        # Ưu tiên gym hot nếu có yêu cầu
        if search_info['hot_search']:
            base_conditions.append("HotResearch = 1")
            search_info['search_type'] = 'hot'
        
        # Xây dựng điều kiện tìm kiếm từ keywords
        search_conditions = []
        valid_keywords = []
        
        for keyword in search_info['keywords']:
            if keyword and len(keyword) >= 2:
                # Escape single quotes để tránh SQL injection
                safe_keyword = keyword.replace("'", "''")
                valid_keywords.append(safe_keyword)
                search_conditions.extend([
                    f"GymName LIKE '%{safe_keyword}%'",
                    f"Address LIKE '%{safe_keyword}%'",
                    f"RepresentName LIKE '%{safe_keyword}%'"
                ])
        
        # Thêm điều kiện địa điểm nếu có
        if search_info['location']:
            safe_location = search_info['location'].replace("'", "''")
            search_conditions.extend([
                f"Address LIKE '%{safe_location}%'",
                f"GymName LIKE '%{safe_location}%'"
            ])
        
        # Xây dựng mệnh đề WHERE
        where_clause = " AND ".join(base_conditions)
        
        if search_conditions:
            keyword_clause = " OR ".join(search_conditions)
            where_clause += f" OR ({keyword_clause})"
        elif not search_info['hot_search']:
            # Nếu không có từ khóa và không phải tìm kiếm hot, return None
            return None
        
        # 5. Tạo SQL query với scoring thông minh
        if valid_keywords:
            primary_keyword = valid_keywords[0]
            sql_query = f"""
            SELECT *, 
                   CASE WHEN HotResearch = 1 THEN 20 ELSE 0 END as hot_score,
                   CASE 
                       WHEN GymName LIKE '%{primary_keyword}%' THEN 30
                       WHEN Address LIKE '%{primary_keyword}%' THEN 25
                       WHEN RepresentName LIKE '%{primary_keyword}%' THEN 15
                       ELSE 5
                   END as relevance_score,
                   CASE 
                       WHEN CreateAt >= DATEADD(year, -1, GETDATE()) THEN 5 
                       ELSE 0 
                   END as recency_score
            FROM dbo.Gym 
            WHERE {where_clause}
            ORDER BY hot_score DESC, relevance_score DESC, recency_score DESC, GymName ASC
            """
        else:
            # Query cho tìm kiếm general hoặc hot gym
            sql_query = f"""
            SELECT *, 
                   CASE WHEN HotResearch = 1 THEN 20 ELSE 0 END as hot_score,
                   10 as relevance_score,
                   CASE 
                       WHEN CreateAt >= DATEADD(year, -1, GETDATE()) THEN 5 
                       ELSE 0 
                   END as recency_score
            FROM dbo.Gym 
            WHERE {where_clause}
            ORDER BY hot_score DESC, relevance_score DESC, recency_score DESC, GymName ASC
            """
        
        print(f"🤖 INTELLIGENT SEARCH: Input='{user_input}' | Type={search_info['search_type']} | Keywords={search_info['keywords'][:3]}")
        return sql_query
        
    except Exception as e:
        print(f"Lỗi trong intelligent_gym_search: {str(e)}")
        return None

def detect_search_intent(user_input):
    """Phát hiện ý định tìm kiếm từ đầu vào của người dùng"""
    user_lower = user_input.lower()
    
    intent_patterns = {
        'location_search': [r'(gần|near|nearby|xung quanh|lân cận|quanh đây)', r'(district \d+|quận \d+|huyện)'],
        'name_search': [r'(tìm .+ gym|gym .+|.+ fitness|.+ center)'],
        'popular_search': [r'(hot|nổi tiếng|phổ biến|được yêu thích|tốt nhất|best)', r'(gym hot|phòng gym hot|fitness hot)'],
        'new_search': [r'(mới|new|vừa mở|recently|gần đây)'],
        'old_search': [r'(cũ|old|lâu năm|uy tín|established)'],
        'price_search': [r'(rẻ|cheap|affordable|giá tốt|budget)'],
        'equipment_search': [r'(thiết bị|equipment|máy tập|facilities)']
    }
    
    detected_intents = []
    for intent, patterns in intent_patterns.items():
        for pattern in patterns:
            if re.search(pattern, user_lower):
                detected_intents.append(intent)
                break
    
    return detected_intents

def classify_query(user_input):
    """Phân loại truy vấn và tạo SQL nếu cần - Được cải thiện"""
    try:
        # 1. Ưu tiên sử dụng intelligent search trước
        intelligent_query = intelligent_gym_search(user_input)
        if intelligent_query:
            print(f"✅ INTELLIGENT_SEARCH: Query generated successfully")
            return True, intelligent_query
        
        # 2. Nếu intelligent search không tạo được query, có nghĩa là:
        # - Câu hỏi không liên quan đến tìm kiếm gym
        # - Hoặc là câu hỏi chào hỏi, tư vấn chung
        
        user_input_lower = user_input.lower()
        
        # 3. Kiểm tra một số trường hợp đặc biệt cuối cùng
        special_gym_cases = [
            'danh sách gym', 'list gym', 'all gym', 'tất cả gym',
            'gym có những gì', 'gym nào', 'which gym'
        ]
        
        if any(case in user_input_lower for case in special_gym_cases):
            # Trường hợp muốn xem tất cả gym
            return True, """
            SELECT *, 
                   CASE WHEN HotResearch = 1 THEN 20 ELSE 0 END as hot_score,
                   10 as relevance_score
            FROM dbo.Gym 
            WHERE Active = 1
            ORDER BY hot_score DESC, GymName ASC
            """
        
        # 4. Nếu không khớp với trường hợp nào -> không cần query database
        print(f"❌ NO_DB_QUERY: '{user_input}' không cần truy vấn database")
        return False, None
        
    except Exception as e:
        print(f"Lỗi trong classify_query: {str(e)}")
        return False, None

def get_response_with_history(user_input, conversation_history=None, longitude=None, latitude=None):
    """Hàm chính để xử lý yêu cầu của người dùng với lịch sử hội thoại"""
    try:
        if conversation_history is None:
            conversation_history = []
        
        conversation_context = build_conversation_context(conversation_history)
        
        # Thêm tin nhắn của người dùng vào lịch sử
        current_conversation = conversation_history.copy()
        current_conversation.append({
            "role": "user",
            "content": sanitize_text_for_json(user_input),
            "timestamp": datetime.now().isoformat()
        })
        
        # Xử lý tìm kiếm gần với tọa độ
        if longitude and latitude and any(keyword in user_input.lower() for keyword in ["gần", "near", "nearby", "xung quanh", "lân cận", "gần đây", "quanh đây"]):
            max_distance = get_nearby_distance_preference(user_input)
            print(f"🎯 SMART RADIUS: User input '{user_input}' → Bán kính được chọn: {max_distance}km")
            
            sql_query = build_nearby_gym_query(longitude, latitude, max_distance)
            results = query_database(sql_query)

            if isinstance(results, str) or not results:
                response_text = f"🤔 Không tìm thấy gym nào trong bán kính {max_distance}km. Hãy thử mở rộng khu vực tìm kiếm!"
                current_conversation.append({
                    "role": "assistant",
                    "content": sanitize_text_for_json(response_text),
                    "timestamp": datetime.now().isoformat()
                })
                return {
                    "promptResponse": sanitize_text_for_json(response_text),
                    "conversation_history": current_conversation
                }

            gyms = [safe_get_row_data(row) for row in results]
            print(f"🔍 DEBUG: Tìm thấy {len(gyms)} gym trong bán kính {max_distance}km")
            for i, gym in enumerate(gyms):
                print(f"  {i+1}. {gym['gymName']} - {gym.get('distance_km', 'N/A')}km")
            
            prompt_response = create_simple_response(gyms, user_input, is_nearby=True)

            current_conversation.append({
                "role": "assistant",
                "content": sanitize_text_for_json(prompt_response),
                "timestamp": datetime.now().isoformat()
            })
            return {
                "gyms": gyms, 
                "promptResponse": sanitize_text_for_json(prompt_response),
                "conversation_history": current_conversation
            }
        
        # Truy vấn cơ sở dữ liệu thông thường
        is_db_query, sql_query = classify_query_with_context(user_input, conversation_context)
        print(f"🔍 QUERY_CLASSIFICATION: is_db_query={is_db_query}, user_input='{user_input}'")
        
        if is_db_query:
            results = query_database(sql_query)
            if isinstance(results, str) or not results:
                response_text = "🤔 Không tìm thấy gym nào phù hợp với tiêu chí của bạn. Hãy thử tìm kiếm khác!"
                current_conversation.append({
                    "role": "assistant",
                    "content": sanitize_text_for_json(response_text),
                    "timestamp": datetime.now().isoformat()
                })
                return {
                    "promptResponse": sanitize_text_for_json(response_text),
                    "conversation_history": current_conversation
                }

            gyms = [safe_get_row_data(row) for row in results]
            print(f"🎯 SEARCH_RESULT: Tìm thấy {len(gyms)} gym từ database")
            
            prompt_response = create_simple_response(gyms, user_input)

            current_conversation.append({
                "role": "assistant",
                "content": sanitize_text_for_json(prompt_response),
                "timestamp": datetime.now().isoformat()
            })
            return {
                "gyms": gyms, 
                "promptResponse": sanitize_text_for_json(prompt_response),
                "conversation_history": current_conversation
            }

        # Hội thoại tự do với Gemini
        enhanced_context = f"""
        Bạn là GymRadar AI - trợ lý tìm kiếm phòng gym thân thiện và chuyên nghiệp tại Việt Nam.
        
        Khả năng:
        - Đưa ra gợi ý phòng gym và thông tin chi tiết
        - Tư vấn thể dục và lịch tập luyện dựa trên mục tiêu
        - Chia sẻ kiến thức về sức khỏe và thể hình
        - Nhớ ngữ cảnh hội thoại để tư vấn nhất quán
        
        Phong cách: Thân thiện, chuyên nghiệp, hiểu biết. Sử dụng emoji phù hợp.
        Luôn kết thúc bằng câu hỏi hoặc gợi ý hành động.
        
        Lịch sử hội thoại:
        {conversation_context}
        """
        
        prompt = f"{enhanced_context}\n\nCâu hỏi của người dùng: {user_input}"
        response = model.generate_content(prompt)
        
        current_conversation.append({
            "role": "assistant", 
            "content": sanitize_text_for_json(response.text),
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "promptResponse": sanitize_text_for_json(response.text),
            "conversation_history": current_conversation
        }

    except Exception as e:
        print(f"Lỗi trong get_response_with_history: {str(e)}")
        error_response = "🤖 Xin lỗi, đã xảy ra lỗi hệ thống. Vui lòng thử lại!"
        
        if 'current_conversation' in locals():
            current_conversation.append({
                "role": "assistant", 
                "content": sanitize_text_for_json(error_response),
                "timestamp": datetime.now().isoformat()
            })
            return {
                "promptResponse": sanitize_text_for_json(error_response),
                "conversation_history": current_conversation
            }
        
        return {"promptResponse": sanitize_text_for_json(error_response)}

def create_simple_response(gyms, user_input, is_nearby=False):
    """Tạo phản hồi đơn giản cho kết quả tìm kiếm gym"""
    if not gyms:
        return "🤔 Không tìm thấy gym nào phù hợp với tiêu chí của bạn."
    
    if len(gyms) == 1:
        gym = gyms[0]
        response = f"🏋️ **{gym['gymName']}**"
        if gym.get('distance_km'):
            response += f" - {format_distance_friendly(gym['distance_km'])}"
        if gym['address']:
            response += f"\n📍 {gym['address']}"
        if gym['hotResearch']:
            response += "\n🔥 Đây là phòng gym rất được yêu thích!"
        return response
    
    elif len(gyms) <= 3:
        response = f"🏋️ **Tìm thấy {len(gyms)} phòng gym:**\n"
        for i, gym in enumerate(gyms, 1):
            name = gym['gymName']
            if gym['hotResearch']:
                name += " 🔥"
            if gym.get('distance_km'):
                name += f" ({format_distance_friendly(gym['distance_km'])})"
            response += f"{i}. **{name}**\n"
        return response
    
    else:
        hot_gyms = [g for g in gyms if g['hotResearch']]
        response = f"🏋️ **Tìm thấy {len(gyms)} phòng gym!**\n"
        
        if hot_gyms:
            response += "🔥 **Các phòng gym phổ biến:**\n"
            for i, gym in enumerate(hot_gyms, 1):
                name = gym['gymName']
                if gym.get('distance_km'):
                    name += f" ({format_distance_friendly(gym['distance_km'])})"
                response += f"{i}. **{name}**\n"
                
            # Hiển thị các gym còn lại
            other_gyms = [g for g in gyms if not g['hotResearch']]
            if other_gyms:
                response += "\n**Các phòng gym khác:**\n"
                for i, gym in enumerate(other_gyms, len(hot_gyms) + 1):
                    name = gym['gymName']
                    if gym.get('distance_km'):
                        name += f" ({format_distance_friendly(gym['distance_km'])})"
                    response += f"{i}. **{name}**\n"
        else:
            response += "**Tất cả phòng gym:**\n"
            for i, gym in enumerate(gyms, 1):
                name = gym['gymName']
                if gym.get('distance_km'):
                    name += f" ({format_distance_friendly(gym['distance_km'])})"
                response += f"{i}. **{name}**\n"
        
        return response

def classify_query_with_context(user_input, conversation_context):
    """Phân loại truy vấn với ngữ cảnh hội thoại"""
    user_input_lower = user_input.lower()
    
    # Danh sách từ khóa chỉ các câu hỏi KHÔNG cần truy vấn database
    non_gym_keywords = [
        # Câu hỏi cá nhân
        'tên tôi', 'tên của tôi', 'lặp lại tên', 'nhắc lại tên', 
        'my name', 'what is my name', 'repeat my name',
        'tôi tên', 'tôi là ai', 'who am i',
        
        # Chào hỏi và lịch sự
        'xin chào', 'hello', 'hi', 'chào bạn', 'hey',
        'cảm ơn', 'thank you', 'thanks', 'cám ơn',
        'tạm biệt', 'bye', 'goodbye', 'chào tạm biệt',
        
        # Tư vấn sức khỏe chung (không cần tìm gym cụ thể)
        'làm sao để', 'cách để', 'how to',
        'ăn gì để', 'what to eat',
        'bài tập nào', 'exercise for',
        'tăng cân', 'giảm cân', 'lose weight', 'gain weight',
        
        # Phản hồi chung
        'ok', 'được rồi', 'tốt', 'good', 'fine', 'đồng ý'
    ]
    
    # Kiểm tra nếu input chứa từ khóa không cần truy vấn DB
    if any(keyword in user_input_lower for keyword in non_gym_keywords):
        return False, None
    
    # Danh sách từ khóa chỉ rõ cần tìm kiếm gym
    gym_search_keywords = [
        'gym', 'fitness', 'thể dục', 'thể hình', 'tập luyện',
        'phòng gym', 'trung tâm', 'center', 'club',
        'tìm', 'search', 'ở đâu', 'where', 'địa chỉ', 'address',
        'gần', 'near', 'nearby', 'quanh', 'xung quanh',
        'quận', 'district', 'thành phố', 'city'
    ]
    
    # Chỉ tiếp tục nếu có từ khóa liên quan đến tìm kiếm gym
    has_gym_intent = any(keyword in user_input_lower for keyword in gym_search_keywords)
    if not has_gym_intent:
        return False, None
    
    detected_intents = detect_search_intent(user_input)
    context_keywords = extract_search_keywords(conversation_context) if conversation_context else []
    
    # Xử lý truy vấn nhận biết ngữ cảnh
    if conversation_context:
        gym_names_in_context = re.findall(r'(\w+\s*(?:gym|fitness|center))', conversation_context.lower())
        
        if gym_names_in_context and any(word in user_input.lower() for word in ['khác', 'other', 'nào khác', 'còn']):
            excluded_gyms = ' AND '.join([f"GymName NOT LIKE '%{name.split()[0]}%'" for name in gym_names_in_context])
            base_query = intelligent_gym_search(user_input)
            if base_query and excluded_gyms:
                enhanced_query = base_query.replace('WHERE Active = 1', f'WHERE Active = 1 AND {excluded_gyms}')
                return True, enhanced_query
    
    # Sử dụng phân loại thông thường
    return classify_query(user_input)

# API Endpoints
@app.post("/chat", summary="Chat với lịch sử hội thoại", response_description="Trả về phản hồi với lịch sử hội thoại")
async def chat_with_history(request: ChatRequest):
    return get_response_with_history(
        user_input=request.prompt,
        conversation_history=request.conversation_history,
        longitude=request.longitude,
        latitude=request.latitude
    )

# Chạy ứng dụng
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
