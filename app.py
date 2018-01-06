from flask import Flask, render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
import dlib
import scipy.misc
import numpy as np
import os
from skimage import io

UPLOAD_FOLDER = './train'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
TOLERANCE = 0.6

def get_face_encodings(filename):
    image = scipy.misc.imread(filename)
    detected_faces = face_detector(image, 1)
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]

def compare_face_encodings(known_faces, face):
    return (np.linalg.norm(known_faces - face, axis=1) <= TOLERANCE)

def find_match(known_faces, names, face):
    matches = compare_face_encodings(known_faces, face)
    count = 0
    for match in matches:
        if match:
            return names[count]
        count += 1
    return 'Not Found'

@app.route("/")
@app.route("/index")
def index():
    image_filenames = filter(lambda x: x.endswith('.npy'), os.listdir('faces/'))
    image_filenames = sorted(image_filenames)
    names = [x[:-4] for x in image_filenames]
    return render_template("index.html",faces=names)
@app.route("/train_page")
def train_page():
    
    return render_template("train.html")

@app.route("/test_page")
def test_page():
    return render_template("test.html")

@app.route('/test', methods=["POST"])
def test():
    face_data = filter(lambda x: x.endswith('.npy'), os.listdir('faces/'))
    face_data = sorted(face_data)
    names = [x[:-4] for x in face_data]
    paths_to_facedata = ['faces/' + x for x in face_data]
    face_encodings = []
    for path in paths_to_facedata:
        face_encodings.append(np.load(path))   
    image = request.files['file']   
    face_encodings_in_image = get_face_encodings(image)
    if len(face_encodings_in_image) != 1:
        print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
        exit()
    match = find_match(face_encodings, names, face_encodings_in_image[0])
    return match

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/train', methods=["POST"])
def train():
    filelist = [f for f in os.listdir('train/')]
    for f in filelist:
        os.remove(os.path.join('train/', f))
    uploaded_files = request.files.getlist("file")
    if request.method == 'POST':
        for f in uploaded_files:
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                #return redirect(url_for('uploaded_file',filename=filename))

        image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('train/'))
        image_filenames = sorted(image_filenames)
        no_of_images = len(image_filenames)
        names = [x[:-4] for x in image_filenames]
        paths = ['train/' + x for x in image_filenames]
        face_encodings = []
        for i in range(0,no_of_images):    
            face_encodings_in_image = get_face_encodings(paths[i])
            if len(face_encodings_in_image) != 1:
                print("Please change image: " + paths[i] + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
                exit()
            np.save('faces/'+names[i],get_face_encodings(paths[i])[0])
    return redirect(url_for('index'))
    
if __name__ == "__main__":
    app.run(debug=True) 