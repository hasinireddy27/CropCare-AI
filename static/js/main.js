document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("uploadForm");

    if (!uploadForm) {
        return;
    }

    const galleryInput = document.getElementById("galleryInput");
    const cameraInput = document.getElementById("cameraInput");

    const galleryButton = document.getElementById("galleryButton");
    const cameraButton = document.getElementById("cameraButton");
    const removeImageButton = document.getElementById(
        "removeImageButton"
    );

    const dropZone = document.getElementById("dropZone");
    const imagePreview = document.getElementById("imagePreview");
    const emptyPreview = document.getElementById("emptyPreview");

    const selectedFileInfo = document.getElementById(
        "selectedFileInfo"
    );
    const selectedFileName = document.getElementById(
        "selectedFileName"
    );

    const detectButton = document.getElementById("detectButton");
    const loadingState = document.getElementById("loadingState");

    const allowedTypes = [
        "image/png",
        "image/jpeg",
        "image/webp"
    ];

    const maximumFileSize = 8 * 1024 * 1024;

    let selectedFile = null;

    galleryButton.addEventListener("click", () => {
        galleryInput.click();
    });

    cameraButton.addEventListener("click", () => {
        cameraInput.click();
    });

    dropZone.addEventListener("click", () => {
        galleryInput.click();
    });

    dropZone.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            galleryInput.click();
        }
    });

    galleryInput.addEventListener("change", () => {
        const file = galleryInput.files[0];

        if (file) {
            handleSelectedFile(file);
        }
    });

    cameraInput.addEventListener("change", () => {
        const file = cameraInput.files[0];

        if (file) {
            handleSelectedFile(file);
        }
    });

    removeImageButton.addEventListener("click", () => {
        clearSelectedImage();
    });

    dropZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropZone.classList.add("drag-active");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-active");
    });

    dropZone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropZone.classList.remove("drag-active");

        const file = event.dataTransfer.files[0];

        if (file) {
            handleSelectedFile(file);
        }
    });

    uploadForm.addEventListener("submit", (event) => {
        if (!selectedFile) {
            event.preventDefault();
            alert("Please select a crop-leaf image.");
            return;
        }

        const transfer = new DataTransfer();
        transfer.items.add(selectedFile);
        galleryInput.files = transfer.files;

        detectButton.disabled = true;
        detectButton.hidden = true;
        loadingState.hidden = false;
    });

    function handleSelectedFile(file) {
        if (!allowedTypes.includes(file.type)) {
            alert("Please select a PNG, JPG, JPEG or WEBP image.");
            clearSelectedImage();
            return;
        }

        if (file.size > maximumFileSize) {
            alert("The selected image must be smaller than 8 MB.");
            clearSelectedImage();
            return;
        }

        selectedFile = file;

        const reader = new FileReader();

        reader.onload = (event) => {
            imagePreview.src = event.target.result;
            imagePreview.hidden = false;
            emptyPreview.hidden = true;
        };

        reader.readAsDataURL(file);

        selectedFileName.textContent =
            `${file.name} · ${formatFileSize(file.size)}`;

        selectedFileInfo.hidden = false;
        detectButton.disabled = false;
        dropZone.classList.add("has-file");
    }

    function clearSelectedImage() {
        selectedFile = null;

        galleryInput.value = "";
        cameraInput.value = "";

        imagePreview.src = "";
        imagePreview.hidden = true;
        emptyPreview.hidden = false;

        selectedFileInfo.hidden = true;
        selectedFileName.textContent = "";

        detectButton.disabled = true;
        dropZone.classList.remove("has-file");
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) {
            return `${bytes} B`;
        }

        if (bytes < 1024 * 1024) {
            return `${(bytes / 1024).toFixed(1)} KB`;
        }

        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }
});