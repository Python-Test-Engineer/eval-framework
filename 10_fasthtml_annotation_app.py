from fasthtml.common import *
import pandas as pd
import csv
import sqlite3
import os
from pathlib import Path

app, rt = fast_app()

# Database setup
DB_FILE = "10_annotations.db"


def init_db():
    """Initialize the SQLite database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create tables
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            item_index INTEGER NOT NULL,
            input_text TEXT NOT NULL,
            output_text TEXT NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets (id),
            UNIQUE(dataset_id, item_index)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id INTEGER NOT NULL,
            is_correct BOOLEAN,
            is_helpful BOOLEAN,
            is_complete BOOLEAN,
            annotated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (item_id) REFERENCES items (id),
            UNIQUE(item_id)
        )
    """
    )

    conn.commit()
    conn.close()


# Global state to store current session info
class AppState:
    def __init__(self):
        self.dataset_id = None
        self.current_index = 0
        self.total_items = 0
        self.filename = None


state = AppState()


def load_csv_to_db(file_path, filename):
    """Load CSV data into database"""
    try:
        # Read pipe-delimited CSV
        df = pd.read_csv(file_path, delimiter="|")
        if "input" not in df.columns or "output" not in df.columns:
            raise ValueError("CSV must contain 'input' and 'output' columns")

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Insert or get dataset
        cursor.execute(
            "INSERT OR IGNORE INTO datasets (filename) VALUES (?)", (filename,)
        )
        cursor.execute("SELECT id FROM datasets WHERE filename = ?", (filename,))
        dataset_id = cursor.fetchone()[0]

        # Clear existing items for this dataset (in case re-uploading)
        cursor.execute("DELETE FROM items WHERE dataset_id = ?", (dataset_id,))

        # Insert items
        for index, row in df.iterrows():
            cursor.execute(
                """
                INSERT INTO items (dataset_id, item_index, input_text, output_text)
                VALUES (?, ?, ?, ?)
            """,
                (dataset_id, index, row["input"], row["output"]),
            )

        conn.commit()
        conn.close()

        # Update state
        state.dataset_id = dataset_id
        state.current_index = 0
        state.total_items = len(df)
        state.filename = filename

        return True
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False


def get_item_data(dataset_id, index):
    """Get item data from database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT i.id, i.input_text, i.output_text, a.is_correct, a.is_helpful, a.is_complete
        FROM items i
        LEFT JOIN annotations a ON i.id = a.item_id
        WHERE i.dataset_id = ? AND i.item_index = ?
    """,
        (dataset_id, index),
    )

    result = cursor.fetchone()
    conn.close()

    if result:
        return {
            "item_id": result[0],
            "input": result[1],
            "output": result[2],
            "is_correct": result[3],
            "is_helpful": result[4],
            "is_complete": result[5],
        }
    return None


def save_annotation_to_db(item_id, is_correct, is_helpful, is_complete):
    """Save annotation to database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Convert string values to proper types
    def parse_bool(value):
        if value == "true":
            return True
        elif value == "false":
            return False
        else:
            return None

    is_correct = parse_bool(is_correct)
    is_helpful = parse_bool(is_helpful)
    is_complete = parse_bool(is_complete)

    cursor.execute(
        """
        INSERT OR REPLACE INTO annotations (item_id, is_correct, is_helpful, is_complete)
        VALUES (?, ?, ?, ?)
    """,
        (item_id, is_correct, is_helpful, is_complete),
    )

    conn.commit()
    conn.close()


def get_annotation_progress(dataset_id):
    """Get progress statistics"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT COUNT(*) FROM items WHERE dataset_id = ?
    """,
        (dataset_id,),
    )
    total = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*) FROM items i
        JOIN annotations a ON i.id = a.item_id
        WHERE i.dataset_id = ?
    """,
        (dataset_id,),
    )
    annotated = cursor.fetchone()[0]

    conn.close()
    return annotated, total


def create_annotation_form(dataset_id, index):
    """Create the annotation form for current item"""
    if not dataset_id:
        return Div("No data loaded", cls="alert alert-warning")

    item_data = get_item_data(dataset_id, index)
    if not item_data:
        return Div("Item not found", cls="alert alert-warning")

    annotated, total = get_annotation_progress(dataset_id)

    return Div(
        H3(f"Item {index + 1} of {total}"),
        # Input/Output Display
        Div(
            H4("Input:"),
            Pre(
                item_data["input"],
                cls="border p-3 bg-light rounded mb-3",
                style="white-space: pre-wrap;",
            ),
            H4("Output:"),
            Pre(
                item_data["output"],
                cls="border p-3 bg-light rounded mb-4",
                style="white-space: pre-wrap;",
            ),
            cls="mb-4",
        ),
        # Annotation Form
        Form(
            Input(type="hidden", name="item_id", value=str(item_data["item_id"])),
            Div(
                H5("Annotations:"),
                # Is Correct
                Div(
                    Label("Is Correct:", cls="form-label fw-bold"),
                    Div(
                        Input(
                            type="radio",
                            name="is_correct",
                            value="true",
                            id="correct_yes",
                            checked=item_data["is_correct"] == 1,
                            cls="form-check-input",
                        ),
                        Label(
                            "Yes", for_="correct_yes", cls="form-check-label ms-1 me-3"
                        ),
                        Input(
                            type="radio",
                            name="is_correct",
                            value="false",
                            id="correct_no",
                            checked=item_data["is_correct"] == 0,
                            cls="form-check-input",
                        ),
                        Label(
                            "No", for_="correct_no", cls="form-check-label ms-1 me-3"
                        ),
                        Input(
                            type="radio",
                            name="is_correct",
                            value="",
                            id="correct_na",
                            checked=item_data["is_correct"] is None,
                            cls="form-check-input",
                        ),
                        Label("N/A", for_="correct_na", cls="form-check-label ms-1"),
                        cls="d-flex align-items-center",
                    ),
                    cls="mb-3",
                ),
                # Is Helpful
                Div(
                    Label("Is Helpful:", cls="form-label fw-bold"),
                    Div(
                        Input(
                            type="radio",
                            name="is_helpful",
                            value="true",
                            id="helpful_yes",
                            checked=item_data["is_helpful"] == 1,
                            cls="form-check-input",
                        ),
                        Label(
                            "Yes", for_="helpful_yes", cls="form-check-label ms-1 me-3"
                        ),
                        Input(
                            type="radio",
                            name="is_helpful",
                            value="false",
                            id="helpful_no",
                            checked=item_data["is_helpful"] == 0,
                            cls="form-check-input",
                        ),
                        Label(
                            "No", for_="helpful_no", cls="form-check-label ms-1 me-3"
                        ),
                        Input(
                            type="radio",
                            name="is_helpful",
                            value="",
                            id="helpful_na",
                            checked=item_data["is_helpful"] is None,
                            cls="form-check-input",
                        ),
                        Label("N/A", for_="helpful_na", cls="form-check-label ms-1"),
                        cls="d-flex align-items-center",
                    ),
                    cls="mb-3",
                ),
                # Is Complete
                Div(
                    Label("Is Complete:", cls="form-label fw-bold"),
                    Div(
                        Input(
                            type="radio",
                            name="is_complete",
                            value="true",
                            id="complete_yes",
                            checked=item_data["is_complete"] == 1,
                            cls="form-check-input",
                        ),
                        Label(
                            "Yes", for_="complete_yes", cls="form-check-label ms-1 me-3"
                        ),
                        Input(
                            type="radio",
                            name="is_complete",
                            value="false",
                            id="complete_no",
                            checked=item_data["is_complete"] == 0,
                            cls="form-check-input",
                        ),
                        Label(
                            "No", for_="complete_no", cls="form-check-label ms-1 me-3"
                        ),
                        Input(
                            type="radio",
                            name="is_complete",
                            value="",
                            id="complete_na",
                            checked=item_data["is_complete"] is None,
                            cls="form-check-input",
                        ),
                        Label("N/A", for_="complete_na", cls="form-check-label ms-1"),
                        cls="d-flex align-items-center",
                    ),
                    cls="mb-4",
                ),
                # Navigation and Save Buttons
                Div(
                    Button(
                        "← Previous",
                        name="action",
                        value="prev",
                        type="submit",
                        disabled=index == 0,
                        cls="btn btn-secondary me-2",
                    ),
                    Button(
                        "Save & Next →",
                        name="action",
                        value="next",
                        type="submit",
                        cls="btn btn-primary me-2",
                    ),
                    Button(
                        "Save",
                        name="action",
                        value="save",
                        type="submit",
                        cls="btn btn-success",
                    ),
                    cls="d-flex gap-2",
                ),
            ),
            method="post",
            action="/annotate",
        ),
        # Progress indicator
        Div(
            f"Progress: {annotated} / {total} annotated ({round(annotated/total*100) if total > 0 else 0}%)",
            cls="mt-3 text-muted small",
        ),
        # Database info
        Div(
            P(
                f"✅ Annotations automatically saved to: {DB_FILE}",
                cls="text-success small mb-0",
            ),
            cls="mt-2",
        ),
        cls="container mt-4",
    )


@rt("/")
def get():
    return Html(
        Head(
            Title("AI Evaluation Annotator"),
            Link(
                href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css",
                rel="stylesheet",
            ),
        ),
        Body(
            Div(
                H1("AI Evaluation Annotation Tool", cls="mb-4"),
                # Database status
                Div(
                    P(
                        f"📊 Database: {DB_FILE} {'✅' if os.path.exists(DB_FILE) else '🔄 (will be created)'}",
                        cls="small text-muted mb-3",
                    ),
                    cls="mb-3",
                ),
                # File upload form
                (
                    Form(
                        Div(
                            Label(
                                "Upload CSV File (pipe-delimited with input|output columns):",
                                cls="form-label",
                            ),
                            Input(
                                type="file",
                                name="csv_file",
                                accept=".csv",
                                cls="form-control mb-3",
                            ),
                            Button("Load CSV", type="submit", cls="btn btn-primary"),
                        ),
                        method="post",
                        action="/upload",
                        enctype="multipart/form-data",
                        cls="mb-4",
                    )
                    if not state.dataset_id
                    else None
                ),
                # Show current file info
                (
                    Div(
                        P(
                            f"Loaded: {state.filename} ({state.total_items} items)"
                            if state.filename
                            else ""
                        ),
                        Div(
                            Button(
                                "Load New File",
                                onclick="location.reload()",
                                cls="btn btn-outline-secondary btn-sm me-2",
                            ),
                            A(
                                "Export Annotations",
                                href="/export",
                                cls="btn btn-outline-primary btn-sm",
                            ),
                            cls="d-flex gap-2",
                        ),
                        cls="mb-3",
                    )
                    if state.dataset_id
                    else None
                ),
                # Annotation interface
                (
                    create_annotation_form(state.dataset_id, state.current_index)
                    if state.dataset_id
                    else None
                ),
                cls="container",
            )
        ),
    )


@rt("/upload", methods=["POST"])
def upload_csv(csv_file: UploadFile):
    """Handle CSV file upload"""
    if csv_file.filename:
        # Save uploaded file temporarily (cross-platform)
        import tempfile

        # Create temp directory if it doesn't exist
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, csv_file.filename)

        with open(temp_path, "wb") as f:
            f.write(csv_file.file.read())

        if load_csv_to_db(temp_path, csv_file.filename):
            return RedirectResponse("/", status_code=303)
        else:
            return Html(
                Head(Title("Error")),
                Body(
                    Div(
                        H2("Error Loading CSV"),
                        P(
                            "Please ensure your CSV file is pipe-delimited and contains 'input' and 'output' columns."
                        ),
                        A("Try Again", href="/", cls="btn btn-primary"),
                        cls="container mt-4",
                    )
                ),
            )
    return RedirectResponse("/", status_code=303)


@rt("/annotate", methods=["POST"])
def save_annotation(
    item_id: int,
    is_correct: str = "",
    is_helpful: str = "",
    is_complete: str = "",
    action: str = "save",
):
    """Save annotation and handle navigation"""
    if not state.dataset_id:
        return RedirectResponse("/", status_code=303)

    # Save annotation to database
    save_annotation_to_db(item_id, is_correct, is_helpful, is_complete)

    # Handle navigation
    if action == "next" and state.current_index < state.total_items - 1:
        state.current_index += 1
    elif action == "prev" and state.current_index > 0:
        state.current_index -= 1

    return RedirectResponse("/", status_code=303)


@rt("/export")
def export_annotations():
    """Export annotations as CSV"""
    if not state.dataset_id:
        return RedirectResponse("/", status_code=303)

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Get all items with annotations for this dataset
    cursor.execute(
        """
        SELECT i.input_text, i.output_text, a.is_correct, a.is_helpful, a.is_complete
        FROM items i
        LEFT JOIN annotations a ON i.id = a.item_id
        WHERE i.dataset_id = ?
        ORDER BY i.item_index
    """,
        (state.dataset_id,),
    )

    results = cursor.fetchall()
    conn.close()

    # Create CSV content
    import io

    csvfile = io.StringIO()
    fieldnames = ["input", "output", "is_correct", "is_helpful", "is_complete"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="|")
    writer.writeheader()

    for row in results:
        writer.writerow(
            {
                "input": row[0],
                "output": row[1],
                "is_correct": row[2],
                "is_helpful": row[3],
                "is_complete": row[4],
            }
        )

    csv_content = csvfile.getvalue()

    return Response(
        csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=annotated_{state.filename}"
        },
    )


if __name__ == "__main__":
    # Initialize database on startup
    init_db()
    import uvicorn

    print("Starting FastHTML Annotation App with SQLite persistence...")
    print("Visit http://localhost:8000 to start annotating")
    print("Export annotations at http://localhost:8000/export")
    print(f"Annotations saved to: {os.path.abspath(DB_FILE)}")
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)
