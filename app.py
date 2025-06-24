import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Khmer Recognition Tool",
    page_icon="🇰🇭",
    layout="wide"
)

# Navigation sidebar
st.sidebar.title("🇰🇭 Khmer Recognition Tool")
app_mode = st.sidebar.selectbox(
    "Choose Recognition Mode:",
    ["🔢 Khmer Digit Recognition", "📝 Khmer Character Recognition"]
)

# --------------------------- Digit Recognition Models and Functions ---------------------------

class DeepLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, label_output_size, correctness_output_size, dropout_rate):
        super(DeepLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

        self.shared_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.shared_fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Two output heads
        self.label_head = nn.Linear(hidden_size // 4, label_output_size)
        self.correctness_head = nn.Linear(hidden_size // 4, correctness_output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.relu(self.shared_fc1(self.dropout(lstm_out[:, -1, :])))
        x = self.relu(self.shared_fc2(self.dropout(x)))

        label_output = self.label_head(x)
        correctness_output = self.correctness_head(x)

        return label_output, correctness_output

def extract_coordinates(json_data):
    temp_list = []
    if json_data is not None:
        objects = json_data.get("objects", [])
        for obj in objects:
            if obj.get("type") == "path" and "path" in obj:
                for point in obj["path"]:
                    if isinstance(point, list):
                        coords = point[1:]
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                x, y = coords[i], coords[i + 1]
                                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                                    temp_list.extend([x, y])
    return temp_list

def scale_coordinates(list_coords):
    x_vals = list_coords[::2]
    y_vals = list_coords[1::2]

    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)

    x_range = max_x - min_x or 1
    y_range = max_y - min_y or 1

    norm_x = [round((x - min_x) / x_range, 8) for x in x_vals]
    norm_y = [round((y - min_y) / y_range, 8) for y in y_vals]

    normalized = list(itertools.chain(*zip(norm_x, norm_y)))
    return normalized

def split_to_substroke(list_coords):
    nested_coords = []
    while len(list_coords) >= 16:
        nested_coords.append(list_coords[:16])
        list_coords = list_coords[16:]
    
    if list_coords:
        padded = list_coords + [0] * (16 - len(list_coords))
        nested_coords.append(padded)

    return nested_coords

def english_to_khmer_digit(digit):
    khmer_digits = {
        0: '០', 1: '១', 2: '២', 3: '៣', 4: '៤',
        5: '៥', 6: '៦', 7: '៧', 8: '៨', 9: '៩'
    }

    if isinstance(digit, list) and len(digit) == 1:
        digit = digit[0]

    try:
        digit_int = int(digit)
        return khmer_digits.get(digit_int, "Invalid input")
    except (ValueError, TypeError):
        return "Invalid input"

def prediction_correctness_to_word(prediction):
    correctness = {0: 'incorrect', 1: 'correct'}

    if isinstance(prediction, list) and len(prediction) == 1:
        prediction = prediction[0]

    try:
        digit_int = int(prediction)
        return correctness.get(digit_int, "Invalid input")
    except (ValueError, TypeError):
        return "Invalid input"

# --------------------------- Character Recognition Models and Functions ---------------------------

# Character mapping
khmer_consonants = [
    "ក", "ខ", "គ", "ឃ", "ង", "ច", "ឆ", "ជ", "ឈ", "ញ",
    "ដ", "ឋ", "ឌ", "ឍ", "ណ", "ត", "ថ", "ទ", "ធ", "ន",
    "ប", "ផ", "ព", "ភ", "ម", "យ", "រ", "ល", "វ",
    "ស", "ហ", "ឡ", "អ",
]
khmer_independent_vowels = [
    "ឥ", "ឦ", "ឧ", "ឩ", "ឪ", "ឫ", "ឬ", "ឭ", "ឮ",
    "ឯ", "ឰ", "ឱ", "ឳ",
]
khmer_dependent_vowels = [
    "ា", "ិ", "ី", "ឹ", "ឺ", "ុ", "ូ", "ួ", "ើ", "ឿ",
    "ៀ", "េ", "ែ", "ៃ", "ោ", "ៅ", "ុំ", "ំ", "ាំ",
    "ះ", "ិះ", "ុះ", "េះ", "ោះ",
]
khmer_symbols = [
    "៖", "។", "៕", "៘", "៉", "៊", "់", "៌", "៍", "៎", "៏", "័", "ឲ", "ៗ", "ៈ"
]
khmer_sub_consonants = ['្ក', '្ខ', '្គ', '្ឃ', '្ង', '្ច', '្ឆ', '្ជ', '្ឈ', '្ញ', '្ឋ', '្ឌ', '្ឍ', '្ណ', '្ត', '្ថ', '្ទ', '្ធ', '្ន', '្ប', '្ផ', '្ព', '្ភ', '្ម', '្យ', '្រ', '្ល', '្វ', '្ស', '្ហ', '្ឡ', '្អ']

unsorted_map = khmer_consonants + khmer_independent_vowels + khmer_dependent_vowels + khmer_symbols + khmer_sub_consonants

if len(unsorted_map) < 119:
    unsorted_map.append("##PLACEHOLDER##")

KHMER_CHARACTER_MAP = sorted(unsorted_map)
DISPLAY_CHARACTERS = [char for char in KHMER_CHARACTER_MAP if char not in ["##PLACEHOLDER##", "!"]]

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden_states):
        energy = torch.tanh(self.attn(hidden_states))
        attn_scores = torch.einsum("bsh,h->bs", energy, self.v)
        return F.softmax(attn_scores, dim=1)

class HybridKhmerRecognizer(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, rnn_hidden_dim, num_layers, num_classes, dropout_prob=0.4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.gru = nn.GRU(
            input_size=cnn_out_channels, hidden_size=rnn_hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0, bidirectional=True
        )
        self.attention = Attention(rnn_hidden_dim * 2)
        self.fc1 = nn.Linear(rnn_hidden_dim * 2, rnn_hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(rnn_hidden_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        cnn_out = self.cnn(x).permute(0, 2, 1)
        gru_out, _ = self.gru(cnn_out)
        attn_weights = self.attention(gru_out)
        context = torch.bmm(attn_weights.unsqueeze(1), gru_out).squeeze(1)
        out = F.relu(self.fc1(context))
        out = self.dropout(out)
        return self.fc2(out)

def preprocess_drawing(json_data, input_dim=16, max_points_per_substroke=8):
    strokes = [obj["path"] for obj in json_data["objects"] if obj["type"] == "path" and obj["path"]]
    if not strokes:
        return None

    raw_points = []
    for i, stroke in enumerate(strokes):
        points = [coord for segment in stroke for coord in segment[1:]]
        raw_points.extend(points)
        if i < len(strokes) - 1:
            raw_points.extend([-1, -1])

    if not raw_points:
        return None

    coords = np.array(raw_points).reshape(-1, 2)
    valid_coords = np.array([c for c in coords if not np.array_equal(c, [-1, -1])])
    
    if valid_coords.shape[0] < 2:
        return None

    min_x, min_y = np.min(valid_coords, axis=0)
    max_x, max_y = np.max(valid_coords, axis=0)

    x_range = max_x - min_x if max_x != min_x else 1
    y_range = max_y - min_y if max_y != min_y else 1

    scaled_list = []
    for x, y in coords:
        if x == -1 and y == -1:
            scaled_list.extend([-1, -1])
        else:
            scaled_x = (x - min_x) / x_range
            scaled_y = (y - min_y) / y_range
            scaled_list.extend([scaled_x, scaled_y])
    
    sub_length = max_points_per_substroke * 2
    substrokes = []
    for i in range(0, len(scaled_list), sub_length):
        chunk = scaled_list[i:i + sub_length]
        if len(chunk) < sub_length:
            chunk += [0] * (sub_length - len(chunk))
        substrokes.append(chunk)

    return torch.tensor(substrokes, dtype=torch.float32).unsqueeze(0)

# --------------------------- App Styling ---------------------------
st.markdown("""
<style>
.prediction-box {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    text-align: center;
    font-size: 1.2rem;
}
.correct {
    background-color: #d4edda;
    border: 2px solid #28a745;
    color: #155724;
}
.incorrect {
    background-color: #f8d7da;
    border: 2px solid #dc3545;
    color: #721c24;
}
.digit-result {
    background-color: #e7f3ff;
    border: 2px solid #007bff;
    color: #004085;
}
</style>
""", unsafe_allow_html=True)

# --------------------------- Main App Logic ---------------------------

if app_mode == "🔢 Khmer Digit Recognition":
    st.title("🔢 Khmer Digit Recognition")
    st.markdown("### Draw a Khmer digit (០-៩) in the canvas below:")
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=5,
        stroke_color="black",
        background_color="white",
        height=400,
        width=600,
        drawing_mode="freedraw",
        key="digit_canvas",
    )
    if st.button("Recognize"):
    # Load digit model
        @st.cache_resource
        def load_digit_model():
            try:
                input_size = 16
                hidden_size = 256
                num_layers = 4
                label_output_size = 10
                correctness_output_size = 2  
                dropout_rate = 0.3

                model = DeepLSTMModel(input_size, hidden_size, num_layers, label_output_size, correctness_output_size, dropout_rate)
                model.load_state_dict(torch.load("best_lstm_model.pth", map_location=torch.device('cpu')))
                model.eval()
                return model
            except FileNotFoundError:
                st.error("Model file 'best_lstm_model.pth' not found!")
                return None
        
        model = load_digit_model()
        
        if model is not None:
            list_coords = extract_coordinates(canvas_result.json_data)

            if list_coords:
                normalized_data = scale_coordinates(list_coords)
                nested_coords = split_to_substroke(normalized_data)
                tensor_data = torch.tensor(nested_coords, dtype=torch.float32)
                tensor_data = torch.stack([tensor_data])

                with torch.no_grad():
                    label_out, correctness_out = model(tensor_data)
                    label_preds = torch.argmax(label_out, dim=1)
                    correct_preds = torch.argmax(correctness_out, dim=1)
                    khmer_number_prediction = english_to_khmer_digit(label_preds)
                    correctness = prediction_correctness_to_word(correct_preds)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="prediction-box digit-result">
                        <h3>🔢 Predicted Digit</h3>
                        <div style='font-size: 4rem;'>{khmer_number_prediction}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    color_class = "correct" if correctness == "correct" else "incorrect"
                    icon = "✅" if correctness == "correct" else "❌"
                    st.markdown(f"""
                    <div class="prediction-box {color_class}">
                        <h3>{icon} Writing Quality</h3>
                        <p style='font-size: 1.5rem;'>{correctness.title()}</p>
                    </div>
                    """, unsafe_allow_html=True)

elif app_mode == "📝 Khmer Character Recognition":
    st.title("📝 Khmer Character Recognition")
    
    target_char = st.selectbox("Select the character you will draw:", DISPLAY_CHARACTERS)
    st.markdown(f"### 🎯 Target Character: **{target_char}**")
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=5,
        stroke_color="black",
        background_color="white",
        height=400,
        width=600,
        drawing_mode="freedraw",
        key="char_canvas",
    )

    if st.button("Recognize"):
        if canvas_result.json_data and canvas_result.json_data["objects"]:
            @st.cache_resource
            def load_char_model():
                try:
                    INPUT_DIM = 16
                    CNN_OUT_CHANNELS = 128
                    RNN_HIDDEN_DIM = 256
                    NUM_LAYERS = 2
                    NUM_CLASSES = 119
                    DROPOUT_PROB = 0.4
                    CHECKPOINT_PATH = "best-checkpoint.ckpt"

                    model = HybridKhmerRecognizer(
                        input_dim=INPUT_DIM, 
                        cnn_out_channels=CNN_OUT_CHANNELS, 
                        rnn_hidden_dim=RNN_HIDDEN_DIM,
                        num_layers=NUM_LAYERS, 
                        num_classes=NUM_CLASSES, 
                        dropout_prob=DROPOUT_PROB
                    )
                    
                    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
                    state_dict = checkpoint.get('state_dict', checkpoint)
                    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                    model.load_state_dict(new_state_dict)
                    model.eval()
                    return model
                except FileNotFoundError:
                    st.error("Model file 'best-checkpoint.ckpt' not found!")
                    return None

            model = load_char_model()
            
            if model is not None:
                processed_input = preprocess_drawing(canvas_result.json_data, input_dim=16)
                
                if processed_input is not None:
                    with torch.no_grad():
                        output = model(processed_input)
                        _, predicted_index = torch.max(output.data, 1)
                        predicted_char = KHMER_CHARACTER_MAP[predicted_index.item()]
                    
                    st.markdown("### 📊 Recognition Result")
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        st.markdown(f"**You drew:**")
                        st.markdown(f"<div style='font-size: 3rem; text-align: center;'>{target_char}</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"**Result:**")
                        if predicted_char == target_char:
                            st.markdown(f"<div style='font-size: 3rem; text-align: center; color: green;'>✅</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='font-size: 3rem; text-align: center; color: red;'>❌</div>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"**Model predicted:**")
                        st.markdown(f"<div style='font-size: 3rem; text-align: center;'>{predicted_char}</div>", unsafe_allow_html=True)
                    
                    if predicted_char == target_char:
                        st.markdown(f"""
                        <div class="prediction-box correct">
                            <h3>🎉 Correct! 🎉</h3>
                            <p>Great job! The model correctly recognized your drawing of <strong>{target_char}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box incorrect">
                            <h3>❌ Incorrect</h3>
                            <p>You intended to draw <strong>{target_char}</strong> but the model predicted <strong>{predicted_char}</strong></p>
                            <p>💡 Try drawing the character more clearly or check the stroke order</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    
                else:
                    st.warning("Not enough drawing data to make a prediction.")
        else:
            st.write("Please draw something on the canvas first.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app combines Khmer digit recognition and character recognition. "
    "Switch between modes using the dropdown above."
)