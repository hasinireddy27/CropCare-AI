from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from services.predictor import CropDiseasePredictor


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp",
}

MAX_FILE_SIZE = 8 * 1024 * 1024


app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

app.config["SECRET_KEY"] = "cropcare-development-secret"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

predictor = CropDiseasePredictor()


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename,
    )


@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "GET":
        return render_template("detect.html")

    image = request.files.get("leaf_image")

    if image is None or image.filename == "":
        flash(
            "Please select a crop-leaf image.",
            "error",
        )
        return redirect(url_for("detect"))

    if not allowed_file(image.filename):
        flash(
            "Only PNG, JPG, JPEG and WEBP images are supported.",
            "error",
        )
        return redirect(url_for("detect"))

    original_name = secure_filename(image.filename)

    extension = original_name.rsplit(
        ".",
        1,
    )[1].lower()

    unique_filename = f"{uuid4().hex}.{extension}"

    saved_path = UPLOAD_FOLDER / unique_filename

    image.save(saved_path)

    try:
        result = predictor.predict(saved_path)

    except ValueError as error:
        saved_path.unlink(
            missing_ok=True
        )

        flash(
            str(error),
            "error",
        )

        return redirect(url_for("detect"))

    return render_template(
        "result.html",
        uploaded_filename=original_name,
        saved_filename=unique_filename,
        result=result,
    )


@app.errorhandler(413)
def file_too_large(error):
    flash(
        "The selected image is too large. Maximum size is 8 MB.",
        "error",
    )

    return redirect(url_for("detect"))


@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False,
    )