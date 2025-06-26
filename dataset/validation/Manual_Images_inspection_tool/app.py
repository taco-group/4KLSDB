from flask import Flask, send_from_directory, render_template, request, jsonify, url_for
import json
import os

app = Flask(__name__)

# Custom route to serve images from your large images folder
@app.route('/hr_images/<path:filename>')
def hr_images(filename):
    images_dir = ''# Update this with the path to your high-resolution images directory
    return send_from_directory(images_dir, filename)

# Home route – will load the main review interface (the grid view)
@app.route('/')
def index():
    # We can load the first page by default; the frontend will call the /get_images endpoint.
    return render_template('index.html')

# Endpoint to update image quality
@app.route('/update_quality', methods=['POST'])
def update_quality():
    image_id = int(request.form.get('id'))
    quality = request.form.get('quality')
    if quality == "null" or quality == "":
        quality = None
    # Load, update, and save the JSON file.
    with open('images.json', 'r') as f:
        images = json.load(f)
    for img in images:
        if img['id'] == image_id:
            img['quality'] = quality
            break
    with open('images.json', 'w') as f:
        json.dump(images, f, indent=4)
    return jsonify(success=True)

# Endpoint to get paginated images
@app.route('/get_images')
def get_images():
    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 50))
    with open('images.json', 'r') as f:
        images = json.load(f)
    # Compute start and end indices for pagination
    start = (page - 1) * limit
    end = start + limit
    paged_images = images[start:end]
    return jsonify(paged_images)

if __name__ == '__main__':
    app.run(debug=True, port=5001)

