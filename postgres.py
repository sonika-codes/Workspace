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
