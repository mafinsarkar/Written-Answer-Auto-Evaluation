from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "mysql+pymysql://t8yupboRkGyjKhk.root:<PASSWORD>@gateway01.ap-southeast-1.prod.aws.tidbcloud.com:4000/test?ssl_ca=<CA_PATH>&ssl_verify_cert=true&ssl_verify_identity=true"

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "ssl": {
            "ca": "isrgrootx1.pem"
        }
    }
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
