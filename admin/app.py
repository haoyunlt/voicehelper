"""
运营后台管理系统 - v1.3.0
提供数据分析、用户管理、内容管理等功能
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from werkzeug.security import generate_password_hash, check_password_hash

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('ADMIN_SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://voicehelper:voicehelper123@postgres:5432/voicehelper')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 获取服务配置
SERVICE_NAME = os.getenv('SERVICE_NAME', os.getenv('ADMIN_SERVICE_NAME', 'voicehelper-admin'))
PORT = int(os.getenv('PORT', os.getenv('ADMIN_PORT', 5001)))
HOST = os.getenv('HOST', '0.0.0.0')

# 初始化扩展
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
CORS(app)


# ==================== 数据模型 ====================

class AdminUser(UserMixin, db.Model):
    """管理员用户"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200))
    role = db.Column(db.String(20), default='viewer')  # viewer, operator, admin, super_admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def has_permission(self, permission):
        role_permissions = {
            'viewer': ['view'],
            'operator': ['view', 'edit'],
            'admin': ['view', 'edit', 'delete'],
            'super_admin': ['view', 'edit', 'delete', 'admin']
        }
        return permission in role_permissions.get(self.role, [])


@login_manager.user_loader
def load_user(user_id):
    return AdminUser.query.get(int(user_id))


# ==================== 路由 ====================

@app.route('/')
@login_required
def dashboard():
    """仪表板首页"""
    # 获取关键指标
    metrics = get_key_metrics()
    
    # 获取图表数据
    charts = {
        'daily_active_users': get_daily_active_users_chart(),
        'conversation_trends': get_conversation_trends_chart(),
        'token_usage': get_token_usage_chart(),
        'response_time': get_response_time_chart()
    }
    
    return render_template('dashboard.html', metrics=metrics, charts=charts)


@app.route('/analytics')
@login_required
def analytics():
    """数据分析页面"""
    # 获取分析数据
    analysis = {
        'user_behavior': analyze_user_behavior(),
        'content_performance': analyze_content_performance(),
        'system_performance': analyze_system_performance(),
        'cost_analysis': analyze_costs()
    }
    
    return render_template('analytics.html', analysis=analysis)


@app.route('/users')
@login_required
def users():
    """用户管理页面"""
    if not current_user.has_permission('view'):
        return redirect(url_for('dashboard'))
    
    # 获取用户列表
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    users_query = db.session.execute("""
        SELECT u.*, 
               COUNT(DISTINCT c.id) as conversation_count,
               COUNT(DISTINCT m.id) as message_count,
               MAX(c.created_at) as last_activity
        FROM users u
        LEFT JOIN conversations c ON u.user_id = c.user_id
        LEFT JOIN messages m ON c.conversation_id = m.conversation_id
        GROUP BY u.id
        ORDER BY u.created_at DESC
        LIMIT :limit OFFSET :offset
    """, {'limit': per_page, 'offset': (page - 1) * per_page})
    
    users_list = users_query.fetchall()
    
    return render_template('users.html', users=users_list, page=page)


@app.route('/conversations')
@login_required
def conversations():
    """会话管理页面"""
    # 获取会话列表
    page = request.args.get('page', 1, type=int)
    status = request.args.get('status', 'all')
    per_page = 20
    
    query = """
        SELECT c.*, u.username, u.nickname,
               COUNT(m.id) as message_count,
               AVG(m.metadata->>'response_time_ms') as avg_response_time
        FROM conversations c
        JOIN users u ON c.user_id = u.user_id
        LEFT JOIN messages m ON c.conversation_id = m.conversation_id
        WHERE 1=1
    """
    
    params = {'limit': per_page, 'offset': (page - 1) * per_page}
    
    if status != 'all':
        query += " AND c.status = :status"
        params['status'] = status
    
    query += """
        GROUP BY c.id, u.username, u.nickname
        ORDER BY c.created_at DESC
        LIMIT :limit OFFSET :offset
    """
    
    conversations_list = db.session.execute(query, params).fetchall()
    
    return render_template('conversations.html', conversations=conversations_list, page=page, status=status)


@app.route('/content')
@login_required
def content():
    """内容管理页面"""
    if not current_user.has_permission('edit'):
        return redirect(url_for('dashboard'))
    
    # 获取文档列表
    documents = db.session.execute("""
        SELECT d.*, 
               COUNT(DISTINCT m.id) as reference_count
        FROM documents d
        LEFT JOIN messages m ON m.references @> jsonb_build_array(jsonb_build_object('doc_id', d.document_id))
        GROUP BY d.id
        ORDER BY d.created_at DESC
    """).fetchall()
    
    return render_template('content.html', documents=documents)


@app.route('/settings')
@login_required
def settings():
    """系统设置页面"""
    if not current_user.has_permission('admin'):
        return redirect(url_for('dashboard'))
    
    # 获取系统配置
    config = {
        'llm_settings': get_llm_settings(),
        'rag_settings': get_rag_settings(),
        'quota_settings': get_quota_settings(),
        'security_settings': get_security_settings()
    }
    
    return render_template('settings.html', config=config)


# ==================== API端点 ====================

@app.route('/api/metrics/realtime')
@login_required
def api_realtime_metrics():
    """实时指标API"""
    metrics = {
        'active_users': get_active_users_count(),
        'active_conversations': get_active_conversations_count(),
        'qps': get_current_qps(),
        'avg_latency': get_avg_latency(),
        'error_rate': get_error_rate()
    }
    return jsonify(metrics)


@app.route('/api/analytics/export')
@login_required
def api_export_analytics():
    """导出分析数据"""
    if not current_user.has_permission('view'):
        return jsonify({'error': 'Permission denied'}), 403
    
    date_from = request.args.get('from', (datetime.now() - timedelta(days=30)).isoformat())
    date_to = request.args.get('to', datetime.now().isoformat())
    format_type = request.args.get('format', 'csv')
    
    # 获取数据
    data = get_analytics_data(date_from, date_to)
    
    if format_type == 'csv':
        df = pd.DataFrame(data)
        csv_data = df.to_csv(index=False)
        return csv_data, 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': f'attachment; filename=analytics_{date_from}_{date_to}.csv'
        }
    elif format_type == 'json':
        return jsonify(data)
    else:
        return jsonify({'error': 'Unsupported format'}), 400


@app.route('/api/users/<user_id>/detail')
@login_required
def api_user_detail(user_id):
    """用户详情API"""
    user_data = db.session.execute("""
        SELECT u.*,
               json_agg(DISTINCT c.*) as conversations,
               json_agg(DISTINCT t.*) as tool_calls
        FROM users u
        LEFT JOIN conversations c ON u.user_id = c.user_id
        LEFT JOIN tool_calls t ON u.user_id = t.user_id
        WHERE u.user_id = :user_id
        GROUP BY u.id
    """, {'user_id': user_id}).fetchone()
    
    if not user_data:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify(dict(user_data))


@app.route('/api/conversations/<conversation_id>/messages')
@login_required
def api_conversation_messages(conversation_id):
    """会话消息API"""
    messages = db.session.execute("""
        SELECT m.*, u.username, u.nickname
        FROM messages m
        LEFT JOIN users u ON m.user_id = u.user_id
        WHERE m.conversation_id = :conversation_id
        ORDER BY m.created_at ASC
    """, {'conversation_id': conversation_id}).fetchall()
    
    return jsonify([dict(m) for m in messages])


@app.route('/api/content/upload', methods=['POST'])
@login_required
def api_upload_content():
    """上传内容API"""
    if not current_user.has_permission('edit'):
        return jsonify({'error': 'Permission denied'}), 403
    
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400
    
    # 保存文件并创建索引任务
    result = process_document_upload(file)
    
    return jsonify(result)


@app.route('/api/settings/update', methods=['POST'])
@login_required
def api_update_settings():
    """更新设置API"""
    if not current_user.has_permission('admin'):
        return jsonify({'error': 'Permission denied'}), 403
    
    settings = request.json
    
    # 更新配置
    update_system_settings(settings)
    
    return jsonify({'success': True, 'message': 'Settings updated'})


# ==================== 辅助函数 ====================

def get_key_metrics():
    """获取关键指标"""
    return {
        'total_users': db.session.execute("SELECT COUNT(*) FROM users").scalar(),
        'total_conversations': db.session.execute("SELECT COUNT(*) FROM conversations").scalar(),
        'total_messages': db.session.execute("SELECT COUNT(*) FROM messages").scalar(),
        'today_active_users': db.session.execute("""
            SELECT COUNT(DISTINCT user_id) FROM conversations 
            WHERE DATE(created_at) = CURRENT_DATE
        """).scalar(),
        'avg_satisfaction': 4.5,  # 示例值
        'system_uptime': '99.9%'  # 示例值
    }


def get_daily_active_users_chart():
    """获取日活用户图表数据"""
    data = db.session.execute("""
        SELECT DATE(created_at) as date, 
               COUNT(DISTINCT user_id) as active_users
        FROM conversations
        WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY DATE(created_at)
        ORDER BY date
    """).fetchall()
    
    df = pd.DataFrame(data, columns=['date', 'active_users'])
    
    fig = px.line(df, x='date', y='active_users', 
                  title='Daily Active Users (Last 30 Days)',
                  labels={'active_users': 'Active Users', 'date': 'Date'})
    
    return fig.to_json()


def get_conversation_trends_chart():
    """获取会话趋势图表"""
    data = db.session.execute("""
        SELECT DATE(created_at) as date,
               COUNT(*) as conversations,
               AVG(EXTRACT(EPOCH FROM (ended_at - started_at))) as avg_duration
        FROM conversations
        WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY DATE(created_at)
        ORDER BY date
    """).fetchall()
    
    df = pd.DataFrame(data, columns=['date', 'conversations', 'avg_duration'])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['date'], y=df['conversations'], name='Conversations'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['avg_duration']/60, 
                            name='Avg Duration (min)', yaxis='y2'))
    
    fig.update_layout(
        title='Conversation Trends',
        yaxis=dict(title='Conversations'),
        yaxis2=dict(title='Duration (min)', overlaying='y', side='right')
    )
    
    return fig.to_json()


def get_token_usage_chart():
    """获取Token使用图表"""
    data = db.session.execute("""
        SELECT DATE(date) as date,
               SUM(CASE WHEN metric_type = 'tokens' THEN metric_value ELSE 0 END) as tokens,
               SUM(CASE WHEN metric_type = 'audio_minutes' THEN metric_value ELSE 0 END) as audio_minutes
        FROM usage_stats
        WHERE date >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY DATE(date)
        ORDER BY date
    """).fetchall()
    
    df = pd.DataFrame(data, columns=['date', 'tokens', 'audio_minutes'])
    
    fig = px.area(df, x='date', y=['tokens', 'audio_minutes'],
                  title='Resource Usage',
                  labels={'value': 'Usage', 'date': 'Date'})
    
    return fig.to_json()


def get_response_time_chart():
    """获取响应时间图表"""
    data = db.session.execute("""
        SELECT DATE_TRUNC('hour', created_at) as hour,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY (metadata->>'response_time_ms')::float) as p50,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY (metadata->>'response_time_ms')::float) as p95,
               percentile_cont(0.99) WITHIN GROUP (ORDER BY (metadata->>'response_time_ms')::float) as p99
        FROM messages
        WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
          AND role = 'assistant'
        GROUP BY DATE_TRUNC('hour', created_at)
        ORDER BY hour
    """).fetchall()
    
    df = pd.DataFrame(data, columns=['hour', 'p50', 'p95', 'p99'])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['hour'], y=df['p50'], name='P50', mode='lines'))
    fig.add_trace(go.Scatter(x=df['hour'], y=df['p95'], name='P95', mode='lines'))
    fig.add_trace(go.Scatter(x=df['hour'], y=df['p99'], name='P99', mode='lines'))
    
    fig.update_layout(
        title='Response Time Percentiles (Last 24 Hours)',
        yaxis=dict(title='Response Time (ms)'),
        xaxis=dict(title='Time')
    )
    
    return fig.to_json()


def analyze_user_behavior():
    """分析用户行为"""
    # 用户留存分析
    retention = db.session.execute("""
        WITH first_use AS (
            SELECT user_id, MIN(DATE(created_at)) as first_date
            FROM conversations
            GROUP BY user_id
        ),
        daily_use AS (
            SELECT user_id, DATE(created_at) as use_date
            FROM conversations
            GROUP BY user_id, DATE(created_at)
        )
        SELECT 
            f.first_date,
            COUNT(DISTINCT f.user_id) as cohort_size,
            COUNT(DISTINCT CASE WHEN d.use_date = f.first_date + INTERVAL '1 day' THEN d.user_id END) as day1,
            COUNT(DISTINCT CASE WHEN d.use_date = f.first_date + INTERVAL '7 days' THEN d.user_id END) as day7,
            COUNT(DISTINCT CASE WHEN d.use_date = f.first_date + INTERVAL '30 days' THEN d.user_id END) as day30
        FROM first_use f
        LEFT JOIN daily_use d ON f.user_id = d.user_id
        GROUP BY f.first_date
        ORDER BY f.first_date DESC
        LIMIT 30
    """).fetchall()
    
    return {
        'retention': retention,
        'user_segments': segment_users(),
        'behavior_patterns': identify_behavior_patterns()
    }


def segment_users():
    """用户分群"""
    segments = db.session.execute("""
        WITH user_stats AS (
            SELECT 
                u.user_id,
                COUNT(DISTINCT c.id) as conversation_count,
                COUNT(DISTINCT DATE(c.created_at)) as active_days,
                AVG(EXTRACT(EPOCH FROM (c.ended_at - c.started_at))) as avg_session_duration
            FROM users u
            LEFT JOIN conversations c ON u.user_id = c.user_id
            WHERE c.created_at >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY u.user_id
        )
        SELECT 
            CASE 
                WHEN conversation_count >= 20 AND active_days >= 15 THEN 'Power Users'
                WHEN conversation_count >= 10 AND active_days >= 7 THEN 'Regular Users'
                WHEN conversation_count >= 1 THEN 'Casual Users'
                ELSE 'Inactive'
            END as segment,
            COUNT(*) as user_count,
            AVG(conversation_count) as avg_conversations,
            AVG(active_days) as avg_active_days
        FROM user_stats
        GROUP BY segment
    """).fetchall()
    
    return segments


def identify_behavior_patterns():
    """识别行为模式"""
    patterns = db.session.execute("""
        SELECT 
            EXTRACT(HOUR FROM created_at) as hour,
            EXTRACT(DOW FROM created_at) as day_of_week,
            COUNT(*) as message_count,
            AVG(LENGTH(content)) as avg_message_length
        FROM messages
        WHERE role = 'user'
          AND created_at >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY EXTRACT(HOUR FROM created_at), EXTRACT(DOW FROM created_at)
        ORDER BY message_count DESC
    """).fetchall()
    
    return patterns


# ==================== 启动应用 ====================

if __name__ == '__main__':
    # 创建数据库表
    with app.app_context():
        db.create_all()
        
        # 创建默认管理员
        if not AdminUser.query.filter_by(username='admin').first():
            admin = AdminUser(
                username='admin',
                email='admin@example.com',
                role='super_admin'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
    
    app.run(host=HOST, port=PORT, debug=True)
