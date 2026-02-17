# streamlit_app.py
# ------------------------------------------------------------
# Sorting Algorithms — Side-by-Side Visual Comparison (Streamlit)
# Enhanced Frontend Edition
# ------------------------------------------------------------

import random
import time
from typing import Generator, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Array generation (same input for both)
# -----------------------------
def make_array(n: int, mode: str, max_val: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    if mode == "random":
        return [rng.randint(1, max_val) for _ in range(n)]
    if mode == "sorted":
        return list(range(1, n + 1))
    if mode == "reverse":
        return list(range(n, 0, -1))
    raise ValueError("mode must be random, sorted, reverse")


# -----------------------------
# Plot (enhanced with better styling)
# -----------------------------
def plot_bars(arr: List[int], highlight: Optional[Tuple[int, int]] = None, title: str = "", algo: str = ""):
    n = len(arr)
    x = list(range(n))

    # Consistent color scheme for all algorithms - professional and visible
    base = "#ddd6fe"      # Light purple - same for all (visible on light background)
    pink = "#c026d3"      # Bright magenta for first highlight
    blue = "#2563eb"      # Bright blue for second highlight

    colors = [base] * n
    if highlight is not None:
        i, j = highlight
        if 0 <= i < n:
            colors[i] = pink
        if 0 <= j < n and j != i:
            colors[j] = blue

    fig = go.Figure(data=[go.Bar(x=x, y=arr, marker_color=colors, marker_line_width=0)])
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=20, family="Inter", weight=700)),
        margin=dict(l=20, r=20, t=55, b=20),
        height=350,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.2)", zeroline=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#1e293b"),
        bargap=0.15,
        showlegend=False,
    )
    return fig


# -----------------------------
# STEP GENERATORS (visual)
# -----------------------------
def bubble_sort_steps(arr: List[int]) -> Generator[Tuple[List[int], Tuple[int, int]], None, None]:
    a = arr[:]
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
                yield a[:], (j, j + 1)
        if not swapped:
            break
    yield a[:], (-1, -1)


def quick_sort_steps(arr: List[int], seed: int) -> Generator[Tuple[List[int], Tuple[int, int]], None, None]:
    """
    Iterative QuickSort with random pivot. Yields on swaps.
    Seed makes pivot choices reproducible per run.
    """
    rng = random.Random(seed)
    a = arr[:]
    stack = [(0, len(a) - 1)]

    def partition_with_yields(low: int, high: int) -> Generator[Tuple[List[int], Tuple[int, int]], None, int]:
        pivot_index = rng.randint(low, high)
        a[pivot_index], a[high] = a[high], a[pivot_index]
        yield a[:], (pivot_index, high)

        pivot = a[high]
        i = low - 1
        for j in range(low, high):
            if a[j] <= pivot:
                i += 1
                if i != j:
                    a[i], a[j] = a[j], a[i]
                    yield a[:], (i, j)

        if i + 1 != high:
            a[i + 1], a[high] = a[high], a[i + 1]
            yield a[:], (i + 1, high)

        return i + 1

    while stack:
        low, high = stack.pop()
        if low >= high:
            continue

        gen = partition_with_yields(low, high)
        try:
            while True:
                state, hl = next(gen)
                yield state, hl
        except StopIteration as e:
            pivot_pos = e.value

        stack.append((low, pivot_pos - 1))
        stack.append((pivot_pos + 1, high))

    yield a[:], (-1, -1)


def merge_sort_steps(arr: List[int]) -> Generator[Tuple[List[int], Tuple[int, int]], None, None]:
    """
    Bottom-up mergesort (no recursion). Yields on writes/copies.
    Highlight is (k,k) for writes.
    """
    a = arr[:]
    n = len(a)
    width = 1
    temp = a[:]

    while width < n:
        for left in range(0, n, 2 * width):
            mid = min(left + width, n)
            right = min(left + 2 * width, n)

            i, j, k = left, mid, left
            while i < mid and j < right:
                if a[i] <= a[j]:
                    temp[k] = a[i]
                    i += 1
                else:
                    temp[k] = a[j]
                    j += 1
                yield temp[:], (k, k)
                k += 1

            while i < mid:
                temp[k] = a[i]
                i += 1
                yield temp[:], (k, k)
                k += 1

            while j < right:
                temp[k] = a[j]
                j += 1
                yield temp[:], (k, k)
                k += 1

            for t in range(left, right):
                a[t] = temp[t]
                yield a[:], (t, t)

        width *= 2

    yield a[:], (-1, -1)


def radix_sort_steps(arr: List[int]) -> Generator[Tuple[List[int], Tuple[int, int]], None, None]:
    """
    LSD radix for non-negative ints. Yields once per digit-pass.
    """
    a = arr[:]
    if any(x < 0 for x in a):
        raise ValueError("Radix here supports non-negative integers only.")
    if not a:
        yield a[:], (-1, -1)
        return

    exp = 1
    max_val = max(a)

    while max_val // exp > 0:
        n = len(a)
        output = [0] * n
        count = [0] * 10

        for x in a:
            digit = (x // exp) % 10
            count[digit] += 1

        for i in range(1, 10):
            count[i] += count[i - 1]

        for i in range(n - 1, -1, -1):
            digit = (a[i] // exp) % 10
            output[count[digit] - 1] = a[i]
            count[digit] -= 1

        a = output[:]
        yield a[:], (-1, -1)
        exp *= 10

    yield a[:], (-1, -1)


def get_generator(algo: str, arr: List[int], seed: int):
    if algo == "Bubble":
        return bubble_sort_steps(arr)
    if algo == "Merge":
        return merge_sort_steps(arr)
    if algo == "Quick":
        return quick_sort_steps(arr, seed=seed)
    if algo == "Radix":
        return radix_sort_steps(arr)
    raise ValueError("Unknown algorithm")


# -----------------------------
# Winner banner helpers
# -----------------------------
def winner_banner(algoA: str, algoB: str) -> Tuple[str, str]:
    """
    Returns (label, message) where label is one of: "info", "success", "warning".
    Uses elapsed time spent stepping as metric (good enough for demo).
    """
    a_done = st.session_state.doneA
    b_done = st.session_state.doneB
    a_t = st.session_state.elapsedA
    b_t = st.session_state.elapsedB

    if a_done and b_done:
        if a_t < b_t:
            return ("success", f"🏆 Winner: Algorithm A ({algoA}) — {a_t:.3f}s vs {b_t:.3f}s")
        if b_t < a_t:
            return ("success", f"🏆 Winner: Algorithm B ({algoB}) — {b_t:.3f}s vs {a_t:.3f}s")
        return ("success", f"🤝 Tie — Both completed in {a_t:.3f}s")

    if a_done and not b_done:
        return ("warning", f"⚡ Algorithm A ({algoA}) finished first — {a_t:.3f}s (B still running)")
    if b_done and not a_done:
        return ("warning", f"⚡ Algorithm B ({algoB}) finished first — {b_t:.3f}s (A still running)")

    if a_t < b_t:
        return ("info", f"📊 Currently: A ({algoA}) is ahead — {a_t:.3f}s vs {b_t:.3f}s")
    if b_t < a_t:
        return ("info", f"📊 Currently: B ({algoB}) is ahead — {b_t:.3f}s vs {a_t:.3f}s")
    return ("info", "⏱️ Race in progress — neck and neck!")


# Algorithm information database
ALGO_INFO = {
    "Bubble": {
        "emoji": "🫧",
        "name": "Bubble Sort",
        "complexity": "O(n²)",
        "best_case": "O(n)",
        "worst_case": "O(n²)",
        "space": "O(1)",
        "stable": "Yes",
        "description": "Repeatedly swaps adjacent elements if they're in wrong order. Simple but inefficient for large datasets.",
        "when_to_use": "Educational purposes, nearly-sorted small datasets",
        "color": "#FF6B9D"
    },
    "Merge": {
        "emoji": "🔀",
        "name": "Merge Sort",
        "complexity": "O(n log n)",
        "best_case": "O(n log n)",
        "worst_case": "O(n log n)",
        "space": "O(n)",
        "stable": "Yes",
        "description": "Divides array in half recursively, then merges sorted halves. Guaranteed O(n log n) performance.",
        "when_to_use": "When stability is required, linked lists, external sorting",
        "color": "#26A69A"
    },
    "Quick": {
        "emoji": "⚡",
        "name": "Quick Sort",
        "complexity": "O(n log n)",
        "best_case": "O(n log n)",
        "worst_case": "O(n²)",
        "space": "O(log n)",
        "stable": "No",
        "description": "Picks a pivot and partitions array around it. Fast in practice due to good cache locality.",
        "when_to_use": "General purpose sorting, when average performance matters most",
        "color": "#FF6F00"
    },
    "Radix": {
        "emoji": "🔢",
        "name": "Radix Sort",
        "complexity": "O(nk)",
        "best_case": "O(nk)",
        "worst_case": "O(nk)",
        "space": "O(n+k)",
        "stable": "Yes",
        "description": "Non-comparison sort that processes digits. Performance depends on number of digits (k).",
        "when_to_use": "Integer sorting with known range, when k is small relative to n",
        "color": "#AB47BC"
    }
}


# -----------------------------
# Enhanced UI with modern design
# -----------------------------
st.set_page_config(
    page_title="Algorithm Visualizer | Sorting Comparison",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

      html, body, [class*="css"]  {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
      }

      .stApp {
        background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 50%, #f0f4ff 100%);
        background-attachment: fixed;
      }

      .main > .block-container {
        padding: 2rem 3rem 3rem 3rem;
        max-width: 1400px;
      }

      /* Hero header */
      .hero-section {
        background: linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(255,255,255,0.95) 100%);
        backdrop-filter: blur(20px);
        padding: 2.5rem 2rem 2rem 2rem;
        border-radius: 24px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(165, 180, 252, 0.3);
      }

      .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
      }

      .hero-subtitle {
        font-size: 1.15rem;
        color: #475569;
        font-weight: 500;
        margin-bottom: 0;
      }

      /* Control panel */
      .control-panel {
        background: rgba(255,255,255,0.98);
        backdrop-filter: blur(20px);
        padding: 1.8rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(165, 180, 252, 0.25);
      }

      .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
      }

      /* Algorithm cards */
      .algo-card {
        background: rgba(255,255,255,0.98);
        backdrop-filter: blur(20px);
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(165, 180, 252, 0.25);
        transition: transform 0.2s, box-shadow 0.2s;
      }

      .algo-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
      }

      .algo-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.8rem;
      }

      .algo-name {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 0.5rem;
      }

      .algo-status {
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.4rem;
      }

      .status-running {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
      }

      .status-paused {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
      }

      .status-done {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
      }

      .algo-stats {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.8rem;
        margin-bottom: 1rem;
      }

      .stat-box {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 0.7rem 1rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
      }

      .stat-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.2rem;
      }

      .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
      }

      /* Buttons */
      .stButton>button {
        border-radius: 12px;
        padding: 0.65rem 1.2rem;
        border: none;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
        transition: all 0.2s;
      }
      
      .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
      }

      /* Selectbox & Slider */
      .stSelectbox > div > div {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        color: #0f172a;
      }

      .stSlider > div > div > div {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
      }

      /* Info boxes */
      .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
      }

      .info-box {
        background: rgba(255,255,255,0.98);
        backdrop-filter: blur(20px);
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(165, 180, 252, 0.25);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.08);
      }

      .info-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
      }

      .info-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
      }

      /* Comparison highlight */
      .comparison-highlight {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(5, 150, 105, 0.08) 100%);
        border: 2px solid rgba(16, 185, 129, 0.25);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        font-weight: 600;
        color: #065f46;
      }

      /* Hide Streamlit branding */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      
      .block-container {
        padding-top: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Session state init
# -----------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "seed" not in st.session_state:
    st.session_state.seed = 42

if "arrA" not in st.session_state:
    st.session_state.arrA = []
if "arrB" not in st.session_state:
    st.session_state.arrB = []
if "genA" not in st.session_state:
    st.session_state.genA = None
if "genB" not in st.session_state:
    st.session_state.genB = None

if "stepsA" not in st.session_state:
    st.session_state.stepsA = 0
if "stepsB" not in st.session_state:
    st.session_state.stepsB = 0

if "elapsedA" not in st.session_state:
    st.session_state.elapsedA = 0.0
if "elapsedB" not in st.session_state:
    st.session_state.elapsedB = 0.0

if "doneA" not in st.session_state:
    st.session_state.doneA = False
if "doneB" not in st.session_state:
    st.session_state.doneB = False

if "hlA" not in st.session_state:
    st.session_state.hlA = None
if "hlB" not in st.session_state:
    st.session_state.hlB = None

# throttle state
if "tick" not in st.session_state:
    st.session_state.tick = 0
if "last_render_time" not in st.session_state:
    st.session_state.last_render_time = 0.0

# render-only state (reduces flicker by updating display less often)
if "renderA" not in st.session_state:
    st.session_state.renderA = []
if "renderB" not in st.session_state:
    st.session_state.renderB = []
if "renderHlA" not in st.session_state:
    st.session_state.renderHlA = None
if "renderHlB" not in st.session_state:
    st.session_state.renderHlB = None


# -----------------------------
# Hero Section
# -----------------------------
st.markdown(
    """
    <div class="hero-section">
        <h1 class="hero-title">⚡ Algorithm Race: Sorting Visualizer</h1>
        <p class="hero-subtitle">
            Compare sorting algorithms side-by-side with real-time visualization. 
            Watch how different strategies tackle the same problem.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Sidebar with Algorithm Info
# -----------------------------
with st.sidebar:
    st.markdown("### 📚 Algorithm Encyclopedia")
    
    selected_info = st.selectbox(
        "Learn about:",
        ["Bubble", "Merge", "Quick", "Radix"],
        key="info_selector"
    )
    
    info = ALGO_INFO[selected_info]
    
    # Display algorithm header
    st.markdown(f"## {info['emoji']} {info['name']}")
    st.markdown(f"**{info['description']}**")
    
    st.markdown("---")
    
    # Complexity metrics
    st.markdown("#### ⏱️ Time Complexity")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average", info['complexity'])
        st.metric("Worst", info['worst_case'])
    with col2:
        st.metric("Best", info['best_case'])
        st.metric("Space", info['space'])
    
    st.markdown("---")
    
    # Additional properties
    st.markdown("#### 📋 Properties")
    st.markdown(f"**Stable:** {info['stable']}")
    
    st.markdown("---")
    
    st.markdown("#### 🎯 Best Used For")
    st.info(info['when_to_use'])
    
    st.markdown("---")
    
    # Python code for each algorithm
    st.markdown("#### 💻 Python Implementation")
    
    code_examples = {
        "Bubble": '''def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr''',
        
        "Merge": '''def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result''',
        
        "Quick": '''def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)''',
        
        "Radix": '''def radix_sort(arr):
    if not arr:
        return arr
    
    max_val = max(arr)
    exp = 1
    
    while max_val // exp > 0:
        counting_sort(arr, exp)
        exp *= 10
    
    return arr

def counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1
    
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1
    
    for i in range(n):
        arr[i] = output[i]'''
    }
    
    st.code(code_examples[selected_info], language="python")
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Tips")
    st.info("""
    **Bubble vs Quick on Reverse:**
    Most dramatic difference!
    
    **Merge vs Quick:**
    Similar speed, different strategies.
    
    **Radix on Large Numbers:**
    Watch it handle digits efficiently.
    """)


# -----------------------------
# Control Panel
# -----------------------------
st.markdown('<div class="control-panel">', unsafe_allow_html=True)
st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    algoA = st.selectbox("🔵 Algorithm A", ["Bubble", "Merge", "Quick", "Radix"], index=0, key="algoA_select")
with col2:
    algoB = st.selectbox("🟠 Algorithm B", ["Bubble", "Merge", "Quick", "Radix"], index=2, key="algoB_select")
with col3:
    mode = st.selectbox("📊 Input Pattern", ["random", "sorted", "reverse"], index=0, key="mode_select")
with col4:
    n = st.slider("📏 Array Size", 10, 200, 60, 10, key="size_slider")

col5, col6 = st.columns(2)
with col5:
    max_val = st.slider("🔢 Maximum Value", 50, 500, 120, 10, key="maxval_slider")
with col6:
    speed = st.slider("⚡ Animation Speed (sec/step)", 0.01, 0.25, 0.03, 0.01, key="speed_slider")

st.markdown("---")

# Advanced controls in expander
with st.expander("🎛️ Advanced Performance Settings"):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        update_every = st.slider("Update display every N steps", 1, 20, 4, 1, key="update_slider")
    with adv_col2:
        max_fps = st.slider("Max FPS", 5, 60, 20, 1, key="fps_slider")

st.markdown("---")

# Control buttons
btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns([1, 1, 1, 1, 1.5])
start_clicked = btn_col1.button("▶️ Start Race", use_container_width=True)
pause_clicked = btn_col2.button("⏸️ Pause", use_container_width=True)
reset_clicked = btn_col3.button("🔄 Reset", use_container_width=True)
new_seed_clicked = btn_col4.button("🎲 New Input", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


def reset_all(new_seed: Optional[int] = None):
    if new_seed is not None:
        st.session_state.seed = new_seed

    st.session_state.running = False
    st.session_state.elapsedA = 0.0
    st.session_state.elapsedB = 0.0
    st.session_state.stepsA = 0
    st.session_state.stepsB = 0
    st.session_state.doneA = False
    st.session_state.doneB = False
    st.session_state.hlA = None
    st.session_state.hlB = None
    st.session_state.tick = 0
    st.session_state.last_render_time = 0.0

    base = make_array(n=n, mode=mode, max_val=max_val, seed=st.session_state.seed)

    st.session_state.arrA = base[:]
    st.session_state.arrB = base[:]
    st.session_state.genA = get_generator(algoA, st.session_state.arrA, seed=st.session_state.seed + 101)
    st.session_state.genB = get_generator(algoB, st.session_state.arrB, seed=st.session_state.seed + 202)

    st.session_state.renderA = st.session_state.arrA[:]
    st.session_state.renderB = st.session_state.arrB[:]
    st.session_state.renderHlA = st.session_state.hlA
    st.session_state.renderHlB = st.session_state.hlB


# reset on first load or param change
params_key = (algoA, algoB, mode, n, max_val)
if "params_key" not in st.session_state:
    st.session_state.params_key = params_key
    reset_all()
elif st.session_state.params_key != params_key:
    st.session_state.params_key = params_key
    reset_all()

if new_seed_clicked:
    reset_all(new_seed=random.randint(1, 10_000))

if reset_clicked:
    reset_all()

if start_clicked:
    st.session_state.running = True

if pause_clicked:
    st.session_state.running = False


# -----------------------------
# LIVE animation (smooth + always continues)
# -----------------------------
should_rerun = False

if st.session_state.running:
    st.session_state.tick += 1

    # cap FPS so it doesn't strobe
    now = time.perf_counter()
    min_frame_time = 1.0 / max_fps
    dt = now - st.session_state.last_render_time
    if dt < min_frame_time:
        time.sleep(min_frame_time - dt)

    # Step A
    if not st.session_state.doneA:
        t0 = time.perf_counter()
        try:
            arr, hl = next(st.session_state.genA)
            st.session_state.arrA = arr
            st.session_state.hlA = hl
            st.session_state.stepsA += 1
        except StopIteration:
            st.session_state.doneA = True
        st.session_state.elapsedA += time.perf_counter() - t0

    # Step B
    if not st.session_state.doneB:
        t0 = time.perf_counter()
        try:
            arr, hl = next(st.session_state.genB)
            st.session_state.arrB = arr
            st.session_state.hlB = hl
            st.session_state.stepsB += 1
        except StopIteration:
            st.session_state.doneB = True
        st.session_state.elapsedB += time.perf_counter() - t0

    # Update DISPLAY only every N steps (reduces flicker)
    if (st.session_state.tick % update_every == 0) or st.session_state.doneA or st.session_state.doneB:
        st.session_state.renderA = st.session_state.arrA[:]
        st.session_state.renderB = st.session_state.arrB[:]
        st.session_state.renderHlA = st.session_state.hlA
        st.session_state.renderHlB = st.session_state.hlB

    # stop when both done
    if st.session_state.doneA and st.session_state.doneB:
        st.session_state.running = False
    else:
        # keep looping
        should_rerun = True

    st.session_state.last_render_time = time.perf_counter()

    # small tick sleep to avoid CPU spikes (separate from fps cap)
    time.sleep(speed)


# -----------------------------
# Dashboard layout
# -----------------------------

# Show live race status banner
label, msg = winner_banner(algoA, algoB)
if st.session_state.running or st.session_state.doneA or st.session_state.doneB:
    if label == "success":
        st.success(msg, icon="🏆")
    elif label == "warning":
        st.warning(msg, icon="⚡")
    else:
        st.info(msg, icon="📊")

st.markdown("---")

left, right = st.columns(2)


def render_algo_card(algo_name: str, steps: int, elapsed: float, done: bool, side: str):
    info = ALGO_INFO[algo_name]
    
    if done:
        status_class = "status-done"
        status_text = "✅ Complete"
    elif st.session_state.running:
        status_class = "status-running"
        status_text = "▶️ Racing"
    else:
        status_class = "status-paused"
        status_text = "⏸️ Paused"
    
    st.markdown(f"""
    <div class="algo-card">
        <div class="algo-header">
            <div class="algo-name">{info['emoji']} {info['name']}</div>
            <div class="algo-status {status_class}">{status_text}</div>
        </div>
        <div class="algo-stats">
            <div class="stat-box">
                <div class="stat-label">Steps Taken</div>
                <div class="stat-value">{steps:,}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Time Elapsed</div>
                <div class="stat-value">{elapsed:.3f}s</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


with left:
    render_algo_card(algoA, st.session_state.stepsA, st.session_state.elapsedA, st.session_state.doneA, "A")
    st.plotly_chart(
        plot_bars(st.session_state.renderA, st.session_state.renderHlA, 
                 title=f"{ALGO_INFO[algoA]['emoji']} {ALGO_INFO[algoA]['name']}", 
                 algo=algoA),
        use_container_width=True,
        config={"displayModeBar": False},
        key="plotly_chart_algorithm_a"
    )


with right:
    render_algo_card(algoB, st.session_state.stepsB, st.session_state.elapsedB, st.session_state.doneB, "B")
    st.plotly_chart(
        plot_bars(st.session_state.renderB, st.session_state.renderHlB, 
                 title=f"{ALGO_INFO[algoB]['emoji']} {ALGO_INFO[algoB]['name']}", 
                 algo=algoB),
        use_container_width=True,
        config={"displayModeBar": False},
        key="plotly_chart_algorithm_b"
    )


# -----------------------------
# Educational comparison section
# -----------------------------
st.markdown("---")

with st.expander("📖 Understanding the Comparison", expanded=True):
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.markdown(f"### {ALGO_INFO[algoA]['emoji']} {ALGO_INFO[algoA]['name']}")
        st.markdown(f"""
        **Complexity:** {ALGO_INFO[algoA]['complexity']}  
        **Space:** {ALGO_INFO[algoA]['space']}  
        **Stable:** {ALGO_INFO[algoA]['stable']}
        
        {ALGO_INFO[algoA]['description']}
        """)
    
    with comp_col2:
        st.markdown(f"### {ALGO_INFO[algoB]['emoji']} {ALGO_INFO[algoB]['name']}")
        st.markdown(f"""
        **Complexity:** {ALGO_INFO[algoB]['complexity']}  
        **Space:** {ALGO_INFO[algoB]['space']}  
        **Stable:** {ALGO_INFO[algoB]['stable']}
        
        {ALGO_INFO[algoB]['description']}
        """)
    
    st.markdown("---")
    
    # Dynamic comparison insights
    if st.session_state.doneA and st.session_state.doneB:
        speed_diff = abs(st.session_state.elapsedA - st.session_state.elapsedB)
        faster = algoA if st.session_state.elapsedA < st.session_state.elapsedB else algoB
        
        if speed_diff > 0.1:
            st.markdown(f"""
            <div class="comparison-highlight">
                💡 <strong>Key Insight:</strong> {ALGO_INFO[faster]['name']} was significantly faster 
                ({speed_diff:.3f}s difference) on this {mode} input of size {n}. 
                This aligns with the theoretical complexity: {ALGO_INFO[faster]['complexity']} 
                often outperforms in this scenario.
            </div>
            """, unsafe_allow_html=True)


# Final winner (when both done)
if st.session_state.doneA and st.session_state.doneB:
    st.markdown("---")
    if st.session_state.elapsedA < st.session_state.elapsedB:
        st.balloons()
        st.success(
            f"🎉 Final Result: {ALGO_INFO[algoA]['emoji']} {ALGO_INFO[algoA]['name']} wins! "
            f"Completed in {st.session_state.elapsedA:.3f}s vs {st.session_state.elapsedB:.3f}s "
            f"({st.session_state.stepsA:,} steps vs {st.session_state.stepsB:,} steps)",
            icon="🏆"
        )
    elif st.session_state.elapsedB < st.session_state.elapsedA:
        st.balloons()
        st.success(
            f"🎉 Final Result: {ALGO_INFO[algoB]['emoji']} {ALGO_INFO[algoB]['name']} wins! "
            f"Completed in {st.session_state.elapsedB:.3f}s vs {st.session_state.elapsedA:.3f}s "
            f"({st.session_state.stepsB:,} steps vs {st.session_state.stepsA:,} steps)",
            icon="🏆"
        )
    else:
        st.success(
            f"🤝 Perfect Tie! Both algorithms completed in {st.session_state.elapsedA:.3f}s",
            icon="⚖️"
        )

# Keep animation going without needing to press Start repeatedly
if should_rerun:
    st.rerun()