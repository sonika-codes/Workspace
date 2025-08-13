from sshtunnel import SSHTunnelForwarder
import psycopg2
import pandas as pd

# SSH connection info (same as DBeaver)
SSH_HOST = "ssh.example.com"
SSH_PORT = 22
SSH_USER = "ssh_user"
SSH_PKEY = "/path/to/private/key"  # or None if using password

# Postgres connection info
PG_HOST = "127.0.0.1"  # Will be localhost once tunneled
PG_PORT = 5432
PG_DB = "my_database"
PG_USER = "db_user"
PG_PASSWORD = "db_password"

# Example DataFrame
df_processed = pd.DataFrame({
    "col1": [1, 2, 3],
    "col2": ["a", "b", "c"]
})

# Open SSH tunnel
with SSHTunnelForwarder(
    (SSH_HOST, SSH_PORT),
    ssh_username=SSH_USER,
    ssh_pkey=SSH_PKEY,  # or ssh_password="your_password"
    remote_bind_address=("127.0.0.1", PG_PORT)
) as tunnel:
    local_port = tunnel.local_bind_port
    print(f"SSH tunnel open on localhost:{local_port}")

    # Connect to Postgres through tunnel
    conn = psycopg2.connect(
        host=PG_HOST,
        port=local_port,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD
    )
    cur = conn.cursor()

    # Insert data row-by-row (can also do bulk insert)
    for _, row in df_processed.iterrows():
        cur.execute(
            "INSERT INTO my_results_table (col1, col2) VALUES (%s, %s)",
            (row.col1, row.col2)
        )

    conn.commit()
    cur.close()
    conn.close()

print("Data inserted successfully!")

from sshtunnel import SSHTunnelForwarder  # Run pip install sshtunnel
from sqlalchemy.orm import sessionmaker  # Run pip install sqlalchemy
from sqlalchemy import create_engine

with SSHTunnelForwarder(
        ('<remote server ip>', 22),  # Remote server IP and SSH port
        ssh_username="<username>",
        ssh_password="<password>",
        remote_bind_address=(
        '<local server ip>', 5432)) as server:  # PostgreSQL server IP and sever port on remote machine

    server.start()  # start ssh sever
    print
    'Server connected via SSH'

    # connect to PostgreSQL
    local_port = str(server.local_bind_port)
    engine = create_engine('postgresql://<username>:<password>@127.0.0.1:' + local_port + '/database_name')

    Session = sessionmaker(bind=engine)
    session = Session()

    print
    'Database session created'

    # test data retrieval
    test = session.execute("SELECT * FROM database_table")
    for row in test:
        print
        row['id']

    session.close()


import paramiko
import socket
import psycopg2
import pandas as pd

# ==== SSH connection details ====
SSH_HOST = "ssh.example.com"
SSH_PORT = 22
SSH_USER = "ssh_user"
SSH_KEY_FILE = "/path/to/private/key"  # Or None if password auth
# SSH_PASSWORD = "your_password"

# ==== Postgres details ====
PG_HOST = "127.0.0.1"  # This is from the SSH host's perspective
PG_PORT = 5432
PG_DB = "my_database"
PG_USER = "db_user"
PG_PASSWORD = "db_password"

# ==== Table to read ====
TABLE_NAME = "my_results_table"

# Step 1: Connect to SSH server
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh_client.connect(
    hostname=SSH_HOST,
    port=SSH_PORT,
    username=SSH_USER,
    key_filename=SSH_KEY_FILE,  # Or password=SSH_PASSWORD
)

# Step 2: Create direct TCP connection from local to remote Postgres
class ForwardedSocket(socket.socket):
    """A socket that talks over Paramiko's Transport."""
    def __init__(self, transport, dest_addr, dest_port):
        super().__init__(socket.AF_INET, socket.SOCK_STREAM)
        chan = transport.open_channel(
            kind="direct-tcpip",
            dest_addr=(dest_addr, dest_port),
            src_addr=("127.0.0.1", 0)
        )
        self.chan = chan
        self.settimeout(10)
        self._sock = chan

# Grab underlying Transport
transport = ssh_client.get_transport()

# Step 3: Connect to Postgres via psycopg2 using the forwarded channel
sock = ForwardedSocket(transport, PG_HOST, PG_PORT)

conn = psycopg2.connect(
    database=PG_DB,
    user=PG_USER,
    password=PG_PASSWORD,
    host="",
    port=0,
    connection_factory=None,
    cursor_factory=None,
    async_=False,
    options=None,
    target_session_attrs=None,
    sock=sock
)

# Step 4: List tables
with conn.cursor() as cur:
    cur.execute("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
          AND table_schema NOT IN ('pg_catalog', 'information_schema');
    """)
    for schema, table in cur.fetchall():
        print(f"{schema}.{table}")

# Step 5: Read a specific table into Pandas
df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} LIMIT 10;", conn)
print("\nSample data:")
print(df)

# Cleanup
conn.close()
ssh_client.close()
