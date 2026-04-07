# FinGent Docker 部署指南

## 项目结构

```
FinGent/
├── docker-compose.yml      # Docker Compose 配置
├── Dockerfile.backend      # 后端 Dockerfile
├── Dockerfile.frontend     # 前端 Dockerfile
├── nginx.conf              # Nginx 配置
├── .dockerignore           # Docker 忽略文件
├── api_server.py           # FastAPI 后端入口
├── main.py                 # CLI 入口
├── requirements.txt        # Python 依赖
├── Frontend/               # Vue3 前端
└── Data/                   # 数据目录
```

## 快速开始

### 1. 环境准备

确保已安装 Docker 和 Docker Compose：

```bash
docker --version
docker-compose --version
```

### 2. 配置环境变量

项目已包含 `.env` 文件，如需修改请编辑：

```bash
# 检查环境变量
cat .env
```

必需的环境变量：
- `QIANWEN_API_KEY` - 通义千问 API Key
- `FMP_API_KEY` - Financial Modeling Prep API Key
- `TIINGO_API_KEY` - Tiingo API Key

### 3. 构建并启动

```bash
# 构建并启动所有服务
docker-compose up --build -d

# 查看日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f backend
docker-compose logs -f frontend
```

### 4. 访问服务

- **前端界面**: http://localhost
- **后端 API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

### 5. 停止服务

```bash
# 停止服务
docker-compose down

# 停止并删除数据卷（谨慎使用）
docker-compose down -v
```

## 常用命令

```bash
# 重新构建特定服务
docker-compose build backend
docker-compose build frontend

# 重启服务
docker-compose restart backend

# 进入容器调试
docker-compose exec backend bash
docker-compose exec mysql mysql -u root -p

# 查看容器状态
docker-compose ps

# 查看资源使用
docker stats
```

## 数据持久化

- **MySQL 数据**: 存储在 Docker Volume `fingent_mysql_data`
- **Redis 数据**: 存储在 Docker Volume `fingent_redis_data`
- **应用数据**: `./Data` 目录挂载到容器

## 端口映射

| 服务 | 容器端口 | 主机端口 |
|------|---------|---------|
| Frontend | 80 | 80 |
| Backend | 8000 | 8000 |
| MySQL | 3306 | 3333 |
| Redis | 6379 | 6379 |

## 生产部署建议

1. **使用 HTTPS**: 配置 SSL 证书
2. **环境变量**: 使用 Docker Secrets 或环境变量管理敏感信息
3. **日志收集**: 配置日志驱动（如 fluentd）
4. **监控**: 添加 Prometheus/Grafana 监控
5. **备份**: 定期备份 MySQL 数据

## 故障排查

### 后端启动失败
```bash
# 检查依赖安装
docker-compose exec backend pip list

# 查看详细日志
docker-compose logs backend --tail=100
```

### 数据库连接失败
```bash
# 检查 MySQL 状态
docker-compose exec mysql mysql -u root -p -e "SHOW DATABASES;"

# 重置数据库（会丢失数据）
docker-compose down -v
docker-compose up -d
```

### 前端构建失败
```bash
# 进入前端容器调试
docker-compose exec frontend sh

# 重新构建前端
docker-compose build --no-cache frontend
```
