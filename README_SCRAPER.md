# Medicine Data Scraper & Updater

## Mô tả
Hệ thống tự động thu thập và cập nhật thông tin thuốc từ các nguồn uy tín (drugs.com), sau đó sử dụng LLM (GPT-4) để dịch sang tiếng Việt và chuẩn hóa dữ liệu.

## Cài đặt

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Cấu hình API Key
Tạo file `.env` từ `.env.example`:
```bash
cp .env.example .env
```

Thêm OpenAI API key vào file `.env`:
