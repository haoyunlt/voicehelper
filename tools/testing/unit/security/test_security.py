"""
安全测试用例
测试覆盖：认证、授权、输入验证、XSS防护、SQL注入防护、敏感数据处理
"""

import pytest
import json
import hashlib
import jwt
import time
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import re

class TestAuthenticationSecurity:
    """认证安全测试"""
    
    def test_jwt_token_validation(self):
        """测试JWT令牌验证"""
        # 有效令牌测试
        secret_key = "test-secret-key"
        payload = {
            "user_id": "123",
            "roles": ["user"],
            "exp": int(time.time()) + 3600  # 1小时后过期
        }
        
        valid_token = jwt.encode(payload, secret_key, algorithm="HS256")
        
        # 验证有效令牌
        try:
            decoded = jwt.decode(valid_token, secret_key, algorithms=["HS256"])
            assert decoded["user_id"] == "123"
            assert "user" in decoded["roles"]
        except jwt.InvalidTokenError:
            pytest.fail("有效令牌验证失败")
    
    def test_jwt_token_expiration(self):
        """测试JWT令牌过期"""
        secret_key = "test-secret-key"
        payload = {
            "user_id": "123",
            "exp": int(time.time()) - 3600  # 1小时前过期
        }
        
        expired_token = jwt.encode(payload, secret_key, algorithm="HS256")
        
        # 验证过期令牌
        with pytest.raises(jwt.ExpiredSignatureError):
            jwt.decode(expired_token, secret_key, algorithms=["HS256"])
    
    def test_jwt_token_tampering(self):
        """测试JWT令牌篡改检测"""
        secret_key = "test-secret-key"
        payload = {"user_id": "123", "exp": int(time.time()) + 3600}
        
        valid_token = jwt.encode(payload, secret_key, algorithm="HS256")
        
        # 篡改令牌
        tampered_token = valid_token[:-5] + "XXXXX"
        
        # 验证篡改检测
        with pytest.raises(jwt.InvalidTokenError):
            jwt.decode(tampered_token, secret_key, algorithms=["HS256"])
    
    def test_password_hashing(self):
        """测试密码哈希安全性"""
        password = "test_password_123"
        salt = "random_salt"
        
        # 使用安全的哈希算法
        hashed = hashlib.pbkdf2_hmac('sha256', 
                                   password.encode('utf-8'), 
                                   salt.encode('utf-8'), 
                                   100000)
        
        # 验证哈希结果
        assert len(hashed) == 32  # SHA256输出长度
        assert hashed != password.encode('utf-8')  # 不应该是明文
        
        # 验证相同输入产生相同哈希
        hashed2 = hashlib.pbkdf2_hmac('sha256', 
                                     password.encode('utf-8'), 
                                     salt.encode('utf-8'), 
                                     100000)
        assert hashed == hashed2
    
    def test_session_security(self):
        """测试会话安全性"""
        # 模拟会话数据
        session_data = {
            "user_id": "123",
            "created_at": time.time(),
            "last_activity": time.time(),
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0..."
        }
        
        # 验证会话超时
        session_timeout = 3600  # 1小时
        current_time = time.time()
        
        # 模拟过期会话
        expired_session = session_data.copy()
        expired_session["last_activity"] = current_time - session_timeout - 1
        
        is_expired = (current_time - expired_session["last_activity"]) > session_timeout
        assert is_expired, "会话应该已过期"
        
        # 验证活跃会话
        active_session = session_data.copy()
        active_session["last_activity"] = current_time - 100  # 100秒前
        
        is_active = (current_time - active_session["last_activity"]) <= session_timeout
        assert is_active, "会话应该仍然活跃"


class TestInputValidationSecurity:
    """输入验证安全测试"""
    
    def test_sql_injection_prevention(self):
        """测试SQL注入防护"""
        # 常见SQL注入攻击模式
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1; DELETE FROM users WHERE 1=1; --",
            "' UNION SELECT * FROM users --"
        ]
        
        def sanitize_sql_input(input_str):
            """简单的SQL输入清理函数"""
            if not isinstance(input_str, str):
                return str(input_str)
            
            # 移除或转义危险字符
            dangerous_patterns = [
                r"'", r'"', r';', r'--', r'/\*', r'\*/',
                r'\bDROP\b', r'\bDELETE\b', r'\bUNION\b',
                r'\bSELECT\b', r'\bINSERT\b', r'\bUPDATE\b'
            ]
            
            cleaned = input_str
            for pattern in dangerous_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            
            return cleaned.strip()
        
        # 测试每个恶意输入
        for malicious_input in malicious_inputs:
            sanitized = sanitize_sql_input(malicious_input)
            
            # 验证危险关键词被移除
            assert 'DROP' not in sanitized.upper()
            assert 'DELETE' not in sanitized.upper()
            assert 'UNION' not in sanitized.upper()
            assert '--' not in sanitized
    
    def test_xss_prevention(self):
        """测试XSS攻击防护"""
        # 常见XSS攻击载荷
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "';alert('XSS');//",
            "<iframe src=javascript:alert('XSS')></iframe>"
        ]
        
        def sanitize_html_input(input_str):
            """HTML输入清理函数"""
            if not isinstance(input_str, str):
                return str(input_str)
            
            # 移除危险的HTML标签和属性
            dangerous_patterns = [
                r'<script[^>]*>.*?</script>',
                r'<iframe[^>]*>.*?</iframe>',
                r'<object[^>]*>.*?</object>',
                r'<embed[^>]*>.*?</embed>',
                r'javascript:',
                r'on\w+\s*=',  # 事件处理器
                r'<svg[^>]*>.*?</svg>'
            ]
            
            cleaned = input_str
            for pattern in dangerous_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
            
            return cleaned
        
        # 测试每个XSS载荷
        for payload in xss_payloads:
            sanitized = sanitize_html_input(payload)
            
            # 验证危险内容被移除
            assert '<script>' not in sanitized.lower()
            assert 'javascript:' not in sanitized.lower()
            assert 'onerror=' not in sanitized.lower()
            assert 'onload=' not in sanitized.lower()
    
    def test_file_upload_validation(self):
        """测试文件上传安全验证"""
        def validate_file_upload(filename, content, max_size=10*1024*1024):
            """文件上传验证函数"""
            errors = []
            
            # 检查文件名
            if not filename or len(filename) > 255:
                errors.append("文件名无效")
            
            # 检查文件扩展名
            allowed_extensions = {'.txt', '.pdf', '.doc', '.docx', '.jpg', '.png'}
            file_ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
            
            if file_ext not in allowed_extensions:
                errors.append(f"不支持的文件类型: {file_ext}")
            
            # 检查文件大小
            if len(content) > max_size:
                errors.append(f"文件大小超过限制: {len(content)} > {max_size}")
            
            # 检查恶意文件名
            dangerous_names = ['..', '/', '\\', 'CON', 'PRN', 'AUX', 'NUL']
            if any(danger in filename.upper() for danger in dangerous_names):
                errors.append("文件名包含危险字符")
            
            return len(errors) == 0, errors
        
        # 测试有效文件
        valid, errors = validate_file_upload("document.pdf", b"PDF content", 1024)
        assert valid, f"有效文件验证失败: {errors}"
        
        # 测试无效扩展名
        valid, errors = validate_file_upload("malware.exe", b"content", 1024)
        assert not valid, "应该拒绝.exe文件"
        
        # 测试文件过大
        valid, errors = validate_file_upload("large.txt", b"x" * (11*1024*1024), 10*1024*1024)
        assert not valid, "应该拒绝过大文件"
        
        # 测试路径遍历攻击
        valid, errors = validate_file_upload("../../../etc/passwd", b"content", 1024)
        assert not valid, "应该拒绝路径遍历攻击"
    
    def test_rate_limiting(self):
        """测试请求频率限制"""
        class RateLimiter:
            def __init__(self, max_requests=100, time_window=3600):
                self.max_requests = max_requests
                self.time_window = time_window
                self.requests = {}  # {ip: [timestamps]}
            
            def is_allowed(self, client_ip):
                current_time = time.time()
                
                # 清理过期记录
                if client_ip in self.requests:
                    self.requests[client_ip] = [
                        ts for ts in self.requests[client_ip]
                        if current_time - ts < self.time_window
                    ]
                else:
                    self.requests[client_ip] = []
                
                # 检查是否超过限制
                if len(self.requests[client_ip]) >= self.max_requests:
                    return False
                
                # 记录当前请求
                self.requests[client_ip].append(current_time)
                return True
        
        # 测试频率限制
        limiter = RateLimiter(max_requests=5, time_window=60)
        client_ip = "192.168.1.100"
        
        # 前5个请求应该被允许
        for i in range(5):
            assert limiter.is_allowed(client_ip), f"第{i+1}个请求应该被允许"
        
        # 第6个请求应该被拒绝
        assert not limiter.is_allowed(client_ip), "第6个请求应该被拒绝"


class TestDataPrivacySecurity:
    """数据隐私安全测试"""
    
    def test_sensitive_data_masking(self):
        """测试敏感数据脱敏"""
        def mask_sensitive_data(data):
            """敏感数据脱敏函数"""
            if not isinstance(data, str):
                return data
            
            # 手机号脱敏
            data = re.sub(r'(\d{3})\d{4}(\d{4})', r'\1****\2', data)
            
            # 身份证号脱敏
            data = re.sub(r'(\d{6})\d{8}(\d{4})', r'\1********\2', data)
            
            # 邮箱脱敏
            data = re.sub(r'(\w{1,3})\w*@(\w+)', r'\1***@\2', data)
            
            # 银行卡号脱敏
            data = re.sub(r'(\d{4})\d{8,12}(\d{4})', r'\1********\2', data)
            
            return data
        
        # 测试手机号脱敏
        phone_text = "我的手机号是13812345678"
        masked = mask_sensitive_data(phone_text)
        assert "138****5678" in masked
        assert "13812345678" not in masked
        
        # 测试身份证号脱敏
        id_text = "身份证号：110101199001011234"
        masked = mask_sensitive_data(id_text)
        assert "110101********1234" in masked
        assert "199001011234" not in masked
        
        # 测试邮箱脱敏
        email_text = "邮箱：user123@example.com"
        masked = mask_sensitive_data(email_text)
        assert "use***@example" in masked
        assert "user123@example.com" not in masked
    
    def test_data_encryption(self):
        """测试数据加密"""
        from cryptography.fernet import Fernet
        
        # 生成密钥
        key = Fernet.generate_key()
        cipher_suite = Fernet(key)
        
        # 测试数据
        sensitive_data = "这是敏感信息：用户密码123456"
        
        # 加密
        encrypted_data = cipher_suite.encrypt(sensitive_data.encode())
        
        # 验证加密结果
        assert encrypted_data != sensitive_data.encode()
        assert len(encrypted_data) > len(sensitive_data)
        
        # 解密验证
        decrypted_data = cipher_suite.decrypt(encrypted_data).decode()
        assert decrypted_data == sensitive_data
    
    def test_secure_logging(self):
        """测试安全日志记录"""
        def secure_log_message(message, level="INFO"):
            """安全日志记录函数"""
            # 移除敏感信息
            sensitive_patterns = [
                (r'password["\s]*[:=]["\s]*([^"\s,}]+)', 'password: [REDACTED]'),
                (r'token["\s]*[:=]["\s]*([^"\s,}]+)', 'token: [REDACTED]'),
                (r'api_key["\s]*[:=]["\s]*([^"\s,}]+)', 'api_key: [REDACTED]'),
                (r'\d{15,19}', '[CARD_NUMBER]'),  # 银行卡号
                (r'\d{3}-?\d{2}-?\d{4}', '[SSN]'),  # 社会保障号
            ]
            
            cleaned_message = message
            for pattern, replacement in sensitive_patterns:
                cleaned_message = re.sub(pattern, replacement, cleaned_message, flags=re.IGNORECASE)
            
            return f"[{level}] {time.strftime('%Y-%m-%d %H:%M:%S')} - {cleaned_message}"
        
        # 测试包含敏感信息的日志
        log_with_password = 'User login: {"username": "admin", "password": "secret123"}'
        secure_log = secure_log_message(log_with_password)
        
        assert "secret123" not in secure_log
        assert "[REDACTED]" in secure_log
        
        # 测试包含令牌的日志
        log_with_token = 'API call with token: abc123def456'
        secure_log = secure_log_message(log_with_token)
        
        assert "abc123def456" not in secure_log
        assert "[REDACTED]" in secure_log


class TestAPISecurityHeaders:
    """API安全头测试"""
    
    def test_security_headers(self):
        """测试安全HTTP头"""
        def get_security_headers():
            """获取推荐的安全HTTP头"""
            return {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
                'Content-Security-Policy': "default-src 'self'",
                'Referrer-Policy': 'strict-origin-when-cross-origin',
                'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
            }
        
        headers = get_security_headers()
        
        # 验证关键安全头存在
        assert 'X-Content-Type-Options' in headers
        assert 'X-Frame-Options' in headers
        assert 'X-XSS-Protection' in headers
        assert 'Strict-Transport-Security' in headers
        assert 'Content-Security-Policy' in headers
        
        # 验证头值正确
        assert headers['X-Content-Type-Options'] == 'nosniff'
        assert headers['X-Frame-Options'] == 'DENY'
        assert 'max-age' in headers['Strict-Transport-Security']
    
    def test_cors_configuration(self):
        """测试CORS配置安全性"""
        def validate_cors_config(cors_config):
            """验证CORS配置安全性"""
            issues = []
            
            # 检查是否允许所有源
            if cors_config.get('allow_origins') == ['*']:
                issues.append("不应该允许所有源 (*)")
            
            # 检查是否允许凭据
            if cors_config.get('allow_credentials') and '*' in cors_config.get('allow_origins', []):
                issues.append("允许凭据时不能使用通配符源")
            
            # 检查允许的方法
            dangerous_methods = ['TRACE', 'CONNECT']
            allowed_methods = cors_config.get('allow_methods', [])
            for method in dangerous_methods:
                if method in allowed_methods:
                    issues.append(f"不应该允许危险的HTTP方法: {method}")
            
            return len(issues) == 0, issues
        
        # 测试安全的CORS配置
        secure_config = {
            'allow_origins': ['https://example.com', 'https://app.example.com'],
            'allow_methods': ['GET', 'POST', 'PUT', 'DELETE'],
            'allow_headers': ['Content-Type', 'Authorization'],
            'allow_credentials': True
        }
        
        is_secure, issues = validate_cors_config(secure_config)
        assert is_secure, f"安全配置验证失败: {issues}"
        
        # 测试不安全的CORS配置
        insecure_config = {
            'allow_origins': ['*'],
            'allow_methods': ['GET', 'POST', 'TRACE'],
            'allow_credentials': True
        }
        
        is_secure, issues = validate_cors_config(insecure_config)
        assert not is_secure, "不安全配置应该被检测出来"
        assert len(issues) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
