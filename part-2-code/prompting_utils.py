import os
import re

def read_schema(schema_path):
    '''
    Read the .schema file
    '''
    # TODO
    if not os.path.exists(schema_path):
        return ""
    with open(schema_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return "\n".join(lines)

def extract_sql_query(response):
    '''
    Extract the SQL query from the model's response
    '''
    # TODO
    if response is None:
        return ""
    text = response.strip()

    # Find SELECT position
    m = re.search(r"(SELECT[\s\S]*)", text, flags=re.IGNORECASE)
    if m:
        sql = m.group(1)
        # end when meet '''
        sql = sql.split("```")[0]
        return sql.strip()
    else:
        # return full text
        return text

def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")