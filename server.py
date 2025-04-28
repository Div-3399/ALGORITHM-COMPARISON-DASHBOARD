import pandas as pd
import shutil
import os
import base64
import io
import time
import psutil  # For memory usage calculation
import seaborn as sns
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from mlxtend.frequent_patterns import apriori, fpgrowth

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # or specify ["http://127.0.0.1:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html at root URL
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/index.html", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# Serve apriori.html
@app.get("/apriori.html", response_class=HTMLResponse)
async def read_apriori():
    with open("apriori.html", "r", encoding="utf-8") as f:
        return f.read()

# Serve fp-growth.html
@app.get("/fp-growth.html", response_class=HTMLResponse)
async def read_fp_growth():
    with open("fp-growth.html", "r", encoding="utf-8") as f:
        return f.read()

# Serve comparison.html
@app.get("/comparison.html", response_class=HTMLResponse)
async def read_comparison():
    with open("comparison.html", "r", encoding="utf-8") as f:
        return f.read()

# Serve style.css
@app.get("/style.css")
async def get_style():
    return FileResponse("style.css")

# Handle CSV upload
@app.post("/upload/") 
async def upload(file: UploadFile = File(...)):
    try:
        # Save the uploaded CSV temporarily
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb+") as f:
            shutil.copyfileobj(file.file, f)

        # Read the file
        df = pd.read_csv(file_location)

        # Preprocessing logic
        df.columns = [c.strip().lower() for c in df.columns]
        transactions = df[df.columns[0]].astype(str).str.split(', ')

        unique_items = sorted(set(item for sublist in transactions for item in sublist))
        binary_encoded = pd.DataFrame(False, index=range(len(transactions)), columns=unique_items)
        for i, items in enumerate(transactions):
            binary_encoded.loc[i, items] = True

        dataset = binary_encoded

        # Function to calculate memory usage
        def get_memory_usage():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)  # Memory in MB

        # Actual Algorithm Work: FP-Growth and Apriori
        def execute_apriori(dataset):
            start_time = time.time()
            memory_before = get_memory_usage()
            frequent_itemsets = apriori(dataset, min_support=0.01, use_colnames=True)
            memory_after = get_memory_usage()
            execution_time = time.time() - start_time
            memory_usage = memory_after - memory_before
            return execution_time, memory_usage, len(frequent_itemsets)

        def execute_fpgrowth(dataset):
            start_time = time.time()
            memory_before = get_memory_usage()
            frequent_itemsets = fpgrowth(dataset, min_support=0.01, use_colnames=True)
            memory_after = get_memory_usage()
            execution_time = time.time() - start_time
            memory_usage = memory_after - memory_before
            return execution_time, memory_usage, len(frequent_itemsets)

        # Get execution time, memory usage, and pattern count for both algorithms
        apriori_time, apriori_memory, apriori_patterns = execute_apriori(dataset)
        fpgrowth_time, fpgrowth_memory, fpgrowth_patterns = execute_fpgrowth(dataset)

        metrics = {
            "Execution Time (s)": [fpgrowth_time, apriori_time],
            "Memory Usage (MB)": [fpgrowth_memory, apriori_memory],
            "Pattern Quality (unique patterns)": [fpgrowth_patterns, apriori_patterns]
        }

        # Create charts and encode them in base64
        images = []

        # Seaborn style for better aesthetics
        sns.set(style="whitegrid")

        def create_chart(title, labels, values):
            fig, ax = plt.subplots(figsize=(8, 5))

            # Bar colors based on your CSS (using your theme colors)
            bar_colors = ['#5a4fcf', '#7b6dff']

            # Plotting the vertical bars
            ax.bar(labels, values, color=bar_colors)

            # Customize title and labels with font adjustments
            ax.set_title(title, fontsize=16, fontweight='bold', color='#333333')
            ax.set_xlabel('Algorithms', fontsize=12, color='#333333')
            ax.set_ylabel('Value', fontsize=12, color='#333333')

            # Customize gridlines and axis
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')
            ax.tick_params(axis='x', labelrotation=45, labelsize=10, colors='#333333')
            ax.tick_params(axis='y', labelsize=10, colors='#333333')

            # Add annotations on top of the bars
            for bar in ax.patches:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha="center", va="bottom", fontsize=10, color='black')

            # Save the plot to a buffer and encode it to base64
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return img_base64

        # Create the bar charts for metrics
        images.append(create_chart("Execution Time (s)", ["FP-Growth", "Apriori"], metrics["Execution Time (s)"]))
        images.append(create_chart("Memory Usage (MB)", ["FP-Growth", "Apriori"], metrics["Memory Usage (MB)"]))
        images.append(create_chart("Pattern Quality", ["FP-Growth", "Apriori"], metrics["Pattern Quality (unique patterns)"]))

        # Scalability chart with line plot
        scalability_sizes = [500, 1000, 2000]
        scalability_metrics = {"FP-Growth": [], "Apriori": []}

        # Measure scalability for different dataset sizes
        for size in scalability_sizes:
            subset = dataset.head(size)

            # FP-Growth scalability
            fpgrowth_time, fpgrowth_memory, _ = execute_fpgrowth(subset)
            scalability_metrics["FP-Growth"].append(fpgrowth_time)

            # Apriori scalability
            apriori_time, apriori_memory, _ = execute_apriori(subset)
            scalability_metrics["Apriori"].append(apriori_time)

        # Create the line chart for scalability
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot the line chart for FP-Growth and Apriori
        ax.plot(scalability_sizes, scalability_metrics["FP-Growth"], label="FP-Growth", color='#5a4fcf', marker='o', linestyle='-', linewidth=2, markersize=6)
        ax.plot(scalability_sizes, scalability_metrics["Apriori"], label="Apriori", color='#7b6dff', marker='o', linestyle='-', linewidth=2, markersize=6)

        # Set chart title and labels
        ax.set_title("Scalability: Execution Time vs Dataset Size", fontsize=16, fontweight='bold', color='#333333')
        ax.set_xlabel("Dataset Size", fontsize=12, color='#333333')
        ax.set_ylabel("Execution Time (s)", fontsize=12, color='#333333')

        # Add grid lines for better visibility
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')

        # Add a legend to identify each line
        ax.legend(loc="upper left", fontsize=12, frameon=False)

        # Customize x and y-axis ticks
        ax.tick_params(axis='x', labelsize=10, colors='#333333')
        ax.tick_params(axis='y', labelsize=10, colors='#333333')

        # Save the plot to a buffer and encode it in base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        scalability_chart_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # Append the scalability chart image to the list of images
        images.append(scalability_chart_base64)

        # Delete the temporary file
        os.remove(file_location)

        return JSONResponse(content={"metrics": metrics, "images": images})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})









