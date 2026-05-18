# app.py
from flask import Flask, render_template, request, session, redirect, url_for, send_file
import os
import pandas as pd
import numpy as np
import joblib
import sqlite3
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "super-secret-key"  # change for production

# Ensure folders exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("Models", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Globals (will be set after preprocess)
dataset = None
X_train = X_test = y_train = y_test = None
trained_models_info = {
    "XGBoost": None,
    "RandomForest": None,
    "DecisionTree": None
}
accuracies = {
    "XGBoost": 0.0,
    "RandomForest": 0.0,
    "DecisionTree": 0.0
}

# ------------------------
# Routes: UI Templates
# ------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/adminlogin')
def AdminLogin():
    return render_template('AdminApp/AdminLogin.html')

@app.route('/AdminAction', methods=['POST'])
def AdminAction():
    username = request.form.get('username', '')
    password = request.form.get('password', '')
    # Basic credentials for demo (change in production)
    if username == 'Admin' and password == 'Admin':
        return render_template("AdminApp/AdminHome.html")
    else:
        return render_template("AdminApp/AdminLogin.html", msg="Login Failed..!!")

@app.route('/AdminHome')
def AdminHome():
    return render_template("AdminApp/AdminHome.html")

@app.route('/Upload')
def Upload():
    return render_template("AdminApp/Upload.html")

# ------------------------
# Upload dataset
# ------------------------
@app.route('/UploadAction', methods=['POST'])
def UploadAction():
    global dataset
    if 'dataset' not in request.files:
        return "No file part in request", 400
    file = request.files['dataset']
    if file.filename == '':
        return "No selected file", 400
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)
    try:
        dataset = pd.read_csv(filepath)
    except Exception as e:
        return f"Failed to read CSV: {e}", 400
    columns = dataset.columns.tolist()
    rows = dataset.head(10).values.tolist()
    return render_template('AdminApp/ViewDataset.html', columns=columns, rows=rows, filename=file.filename)

# ------------------------
# Preprocess route (encodes + scales + split)
# ------------------------
@app.route('/preprocess')
def preprocess():
    global dataset, X_train, X_test, y_train, y_test

    # If dataset not uploaded, try default Dataset/Book1.csv
    if dataset is None:
        default_path = "Dataset/Book1.csv"
        if os.path.exists(default_path):
            dataset = pd.read_csv(default_path)
        else:
            return "No dataset uploaded and default Dataset/dataset_sdn.csv missing. Upload a CSV first.", 400

    # drop NaNs
    dataset.dropna(inplace=True)

    # Ensure expected columns exist (adjust if your CSV differs)
    # We'll select required columns if they exist, otherwise bail with message
    required = ['switch', 'src', 'dst', 'pktcount', 'bytecount',
                'dur', 'flows', 'pktrate', 'Protocol', 'tot_kbps', 'Label']
    missing = [c for c in required if c not in dataset.columns]
    if missing:
        return f"Dataset missing required columns: {missing}", 400

    # Prepare label encoders for categorical columns
    categorical_columns = ['switch', 'src', 'dst', 'Protocol']
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        # Fit on full dataset
        dataset[col] = le.fit_transform(dataset[col].astype(str))
        encoders[col] = le

    # Standardize numeric columns (fit scaler on full dataset features)
    numeric_cols = ['pktcount', 'bytecount', 'dur', 'flows', 'pktrate', 'tot_kbps']
    scaler = StandardScaler()
    dataset[numeric_cols] = scaler.fit_transform(dataset[numeric_cols])

    # Select final columns and split
    selected_columns = ['switch', 'src', 'dst', 'pktcount', 'bytecount',
                        'dur', 'flows', 'pktrate', 'Protocol', 'tot_kbps', 'Label']
    df = dataset[selected_columns].copy()

    X = df.drop(columns=['Label'])
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save encoders and scaler for inference
    joblib.dump(encoders, "Models/encoders.pkl")
    joblib.dump(scaler, "Models/scaler.pkl")

    return render_template('AdminApp/SplitStatus.html', total=len(X), train=len(X_train), test=len(X_test))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@app.route('/RandomForest')
def RandomForest():
    global trained_models_info, accuracies, X_train, X_test, y_train, y_test
    if X_train is None:
        return "Please run preprocess first.", 400

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "Models/RFModel.joblib")

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred) * 100
    prec = precision_score(y_test, pred, average='macro') * 100
    rec = recall_score(y_test, pred, average='macro') * 100
    f1 = f1_score(y_test, pred, average='macro') * 100

    accuracies['RandomForest'] = round(acc, 2)
    trained_models_info['RandomForest'] = "Models/RFModel.joblib"

    return render_template('AdminApp/AlgorithmStatus.html',
                           msg="Random Forest Model Generated Successfully..!!",
                           Accuracy = str(round(acc, 2)),
                           Precision = str(round(prec, 2)),
                           Recall = str(round(rec, 2)),
                           F1Score = str(round(f1, 2))
                           )

# ------------------------
# Train Decision Tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# ------------------------
# Train LightGBM
# ------------------------
@app.route('/LGBM')
def LGBM():
    global trained_models_info, accuracies, X_train, X_test, y_train, y_test
    if X_train is None:
        return "Please run preprocess first.", 400

    import lightgbm as lgb
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "Models/LGBMModel.joblib")

    pred = model.predict(X_test)

    acc1 = accuracy_score(y_test, pred) * 100
    prec = precision_score(y_test, pred, average='macro') * 100
    rec = recall_score(y_test, pred, average='macro') * 100
    f1 = f1_score(y_test, pred, average='macro') * 100

    accuracies['LGBM'] = round(acc1, 2)
    trained_models_info['LGBM'] = "Models/LGBMModel.joblib"

    return render_template('AdminApp/AlgorithmStatus.html',
                           msg="LightGBM Model Generated Successfully..!!",
                           Accuracy = str(round(acc1,2)),
                           Precision = str(round(prec,2)),
                           Recall = str(round(rec,2)),
                           F1Score = str(round(f1,2))
                           )


# ------------------------
# Train CatBoost
# ------------------------
@app.route('/CatBoost')
def CatBoost():
    global trained_models_info, accuracies, X_train, X_test, y_train, y_test
    if X_train is None:
        return "Please run preprocess first.", 400

    from catboost import CatBoostClassifier
    model = CatBoostClassifier(verbose=0, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "Models/CatBoostModel.joblib")

    pred = model.predict(X_test)

    acc2 = accuracy_score(y_test, pred) * 100
    prec = precision_score(y_test, pred, average='macro') * 100
    rec = recall_score(y_test, pred, average='macro') * 100
    f1 = f1_score(y_test, pred, average='macro') * 100

    accuracies['CatBoost'] = round(acc2, 2)
    trained_models_info['CatBoost'] = "Models/CatBoostModel.joblib"

    return render_template('AdminApp/AlgorithmStatus.html',
                           msg="CatBoost Model Generated Successfully..!!",
                           Accuracy = str(round(acc2,2)),
                           Precision = str(round(prec,2)),
                           Recall = str(round(rec,2)),
                           F1Score = str(round(f1,2))
                           )
import matplotlib.pyplot as plt
import os

@app.route('/comparison')
def comparison():
    global accuracies

    if not accuracies:
        return "No models have been trained yet!", 400

    # Allow only RF, LGBM, CatBoost
    allowed_models = ["RandomForest", "LGBM", "CatBoost"]
    models = [m for m in accuracies.keys() if m in allowed_models]
    scores = [accuracies[m] for m in models]

    # Create bar chart
    plt.figure(figsize=(6,4))
    plt.bar(models, scores)
    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy (%)")
    plt.title("Algorithm Accuracy Comparison")
    plt.ylim(0, 100)

    # Save chart
    graph_path = "static/comparison.png"
    plt.savefig(graph_path)
    plt.close()

    return render_template("AdminApp/Grpah.html", graph="static/comparison.png")

# User routes (login/register/simple flow)
# ------------------------
@app.route('/userlogin')
def userlogin():
    return render_template('UserApp/Login.html')

@app.route('/register')
def register():
    return render_template('UserApp/Register.html')

@app.route('/RegAction', methods=['POST'])
def RegAction():
    name = request.form.get('name')
    email = request.form.get('email')
    mobile = request.form.get('mobile')
    username = request.form.get('username')
    password = request.form.get('password')

    con = sqlite3.connect('database.db')
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS user(name TEXT, email TEXT, mobile TEXT, username TEXT, password TEXT)")
    cur.execute("SELECT * FROM user WHERE username=? AND password=?", (username, password))
    data = cur.fetchone()
    if data is None:
        cur.execute("INSERT INTO user VALUES (?, ?, ?, ?, ?)", (name, email, mobile, username, password))
        con.commit()
        con.close()
        return render_template('UserApp/Register.html', msg="Successfully Registered..!!")
    else:
        con.close()
        return render_template('UserApp/Register.html', msg="Username and password already exist..!!")

@app.route('/UserAction', methods=['POST'])
def UserAction():
    username = request.form.get('username')
    password = request.form.get('password')
    con = sqlite3.connect('database.db')
    cur = con.cursor()
    cur.execute("SELECT * FROM user WHERE username=? AND password=?", (username, password))
    data = cur.fetchone()
    con.close()
    if data is None:
        return render_template('UserApp/Login.html', msg="Login Failed..!!")
    else:
        session['username'] = data[3]
        return render_template('UserApp/Home.html', username=session['username'])

@app.route('/Detect')
def Detect():
    return render_template('UserApp/Detect.html')

@app.route('/UserHome')
def UserHome():
    return render_template('UserApp/Home.html')

# ------------------------
# DetectAction (inference)
# Uses RandomForest by default. If not present, falls back to Decision Tree or XGBoost if available.
# ------------------------
from flask import Flask, request, render_template
import os
import pandas as pd
import numpy as np
import joblib


@app.route('/DetectAction', methods=['POST'])

def DetectAction():

    # -------------------------------
    # 1. Get form data
    # -------------------------------
    switch = request.form['switch']
    src = request.form['src']
    dst = request.form['dst']
    pktcount = float(request.form['pktcount'])
    bytecount = float(request.form['bytecount'])
    dur = float(request.form['dur'])
    flows = float(request.form['flows'])
    pktrate = float(request.form['pktrate'])
    Protocol = request.form['Protocol'].upper()
    tot_kbps = float(request.form['tot_kbps'])

    # -------------------------------
    # 2. Load model
    # -------------------------------
    if os.path.exists("Models/RFModel.joblib"):
        model_path = "Models/RFModel.joblib"
    elif os.path.exists("Models/LGBMModel.joblib"):
        model_path = "Models/LGBMModel.joblib"
    else:
        model_path = "Models/CatBoostModel.joblib"

    model = joblib.load(model_path)
    encoders = joblib.load("Models/encoders.pkl")
    scaler = joblib.load("Models/scaler.pkl")

    # -------------------------------
    # 3. Create DataFrame
    # -------------------------------
    test = pd.DataFrame([{
        'switch': switch,
        'src': src,
        'dst': dst,
        'pktcount': pktcount,
        'bytecount': bytecount,
        'dur': dur,
        'flows': flows,
        'pktrate': pktrate,
        'Protocol': Protocol,
        'tot_kbps': tot_kbps
    }])

    # -------------------------------
    # 4. Handle categorical features
    # -------------------------------
    cat_cols = ['switch', 'src', 'dst', 'Protocol']

    for col in cat_cols:
        if 'unknown' not in encoders[col].classes_:
            encoders[col].classes_ = np.append(encoders[col].classes_, 'unknown')

        test[col] = test[col].apply(
            lambda x: x if x in encoders[col].classes_ else 'unknown'
        )

        test[col] = encoders[col].transform(test[col])

    # -------------------------------
    # 5. Scale numerical features
    # -------------------------------
    num_cols = ['pktcount', 'bytecount', 'dur', 'flows', 'pktrate', 'tot_kbps']
    test[num_cols] = scaler.transform(test[num_cols])

    # -------------------------------
    # 6. Prediction (SAFE HANDLING)
    # -------------------------------
    pred = model.predict(test)[0]
    prob = model.predict_proba(test)[0]

    # If model returns string labels
    if isinstance(pred, str):
        label = pred
    else:
        label = model.classes_[pred]

    probs = dict(zip(model.classes_, prob))

    print("Prediction:", label)
    print("Probabilities:", probs)

    # -------------------------------
    # 7. Status mapping
    # -------------------------------
    if label == "Benign":
        status = "✅ Normal Traffic (Benign)"
    elif label == "DDoS-ACK":
        status = "🚨 DDoS Attack Detected (ACK Flood)"
    else:
        status = "🚨 DDoS Attack Detected (PSH-ACK Flood)"

    # -------------------------------
    # 8. Render result
    # -------------------------------
    return render_template(
        "UserApp/Result.html",
        status=status,
        Benign=f"{probs.get('Benign', 0):.4f}",
        Ack=f"{probs.get('DDoS-ACK', 0):.4f}",
        PshAck=f"{probs.get('DDoS-PSH-ACK', 0):.4f}"
    )



@app.route('/download_example')
def download_example():
    example_path = "Dataset/Book1.csv"
    if os.path.exists(example_path):
        return send_file(example_path, as_attachment=True)
    else:
        return "No example dataset found.", 404

if __name__ == '__main__':
    app.run(debug=True)
