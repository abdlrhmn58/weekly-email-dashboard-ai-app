import gradio as gr
import time
import requests
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Config ---
API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TOKEN = os.getenv("HF_TOKEN") 
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
REPHRASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# --- Weekly state ---
weekly_tasks = []
task_target = 10

# --- Helper Functions ---
def local_rephrase(text: str) -> str:
    t = text.strip().rstrip(".")
    return f"Completed: {t.capitalize()}, contributing to team objectives."

def hf_rephrase(text: str) -> str:
    prompt = f"Rewrite this goal as exactly one professional achievement sentence, no explanations, no echo:\n{text}"
    try:
        payload = {
            "model": REPHRASE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 64
        }
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and len(data["choices"]) > 0:
            raw = data["choices"][0]["message"]["content"].strip()
            sentences = [s.strip() for s in raw.replace("\n", " ").split(".") if s.strip()]
            if sentences:
                return sentences[0] + "."
            return raw
        return local_rephrase(text)
    except Exception as e:
        print(f"AI Error: {e}")
        return local_rephrase(text)

def generate_professional_email():
    """Generate a professional email from all weekly tasks"""
    if not weekly_tasks:
        return "No tasks to generate email from. Please add achievements first."
    
    achievements_list = "\n".join([f"• {task['rephrased']}" for task in weekly_tasks])
    
    prompt = f"""Write a professional weekly status update email based on these achievements:

{achievements_list}

Format it as a complete email with:
- Professional greeting
- Brief introduction
- Bullet points of achievements
- Forward-looking conclusion
- Professional closing

Keep it concise and professional."""
    
    try:
        payload = {
            "model": REPHRASE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 500
        }
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"].strip()
        return "Error generating email."
    except Exception as e:
        print(f"Email generation error: {e}")
        return f"""Subject: Weekly Status Update - {time.strftime('%B %d, %Y')}

Dear Team,

I wanted to share a quick update on this week's accomplishments:

{achievements_list}

Looking forward to continuing this momentum next week.

Best regards"""

def get_weekly_progress_gauge():
    """Create a comprehensive gauge chart showing weekly progress"""
    completed = len(weekly_tasks)
    pct = 0 if task_target == 0 else min((completed / task_target) * 100, 100)
    
    # Determine color based on progress
    if pct < 40:
        color = "#FF5252"  # Red
    elif pct < 70:
        color = "#FFA726"  # Orange
    else:
        color = "#66BB6A"  # Green
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=completed,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Weekly Progress<br><span style='font-size:0.8em'>Target: {task_target} tasks</span>", 'font': {'size': 24}},
        delta={'reference': task_target, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, task_target], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, task_target * 0.4], 'color': '#FFE6E6'},
                {'range': [task_target * 0.4, task_target * 0.7], 'color': '#FFF4E6'},
                {'range': [task_target * 0.7, task_target], 'color': '#E8F5E9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': task_target
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(t=80, b=40, l=40, r=40),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def get_category_breakdown():
    """Create a donut chart showing task categories"""
    if not weekly_tasks:
        fig = go.Figure()
        fig.add_annotation(
            text="No tasks yet<br>Add your first achievement!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(height=400, margin=dict(t=50, b=50, l=50, r=50))
        return fig
    
    categories = {"Data": 0, "Development": 0, "Meetings": 0, "Documentation": 0, "Other": 0}
    
    for task in weekly_tasks:
        text = task['text'].lower()
        if any(word in text for word in ['data', 'metric', 'chart', 'plot', 'analysis']):
            categories["Data"] += 1
        elif any(word in text for word in ['code', 'dev', 'build', 'fix', 'deploy']):
            categories["Development"] += 1
        elif any(word in text for word in ['meet', 'call', 'discuss', 'presentation']):
            categories["Meetings"] += 1
        elif any(word in text for word in ['doc', 'write', 'report', 'documentation']):
            categories["Documentation"] += 1
        else:
            categories["Other"] += 1
    
    # Filter out zero values
    filtered_cats = {k: v for k, v in categories.items() if v > 0}
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig = go.Figure(data=[go.Pie(
        labels=list(filtered_cats.keys()),
        values=list(filtered_cats.values()),
        hole=0.4,
        marker=dict(colors=colors[:len(filtered_cats)]),
        textinfo='label+percent+value',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Task Categories Breakdown",
        height=400,
        margin=dict(t=80, b=40, l=40, r=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def format_list():
    if not weekly_tasks:
        return "📝 No tasks yet. Add your first achievement above!"
    
    lines = []
    for i, item in enumerate(weekly_tasks, 1):
        lines.append(f"{'='*60}")
        lines.append(f"Task #{i}")
        lines.append(f"Original: {item['text']}")
        lines.append(f"Professional: {item['rephrased']}")
        lines.append(f"Time: {item['added_at']}")
        lines.append("")
    return "\n".join(lines)

def render_progress():
    completed = len(weekly_tasks)
    pending = max(task_target - completed, 0)
    pct = 0 if task_target == 0 else min(round((completed / task_target) * 100), 100)
    
    return f"""
    <div style="font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif; max-width: 560px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <h3 style="margin:0; font-size:1.2em;">📊 Weekly Progress</h3>
        <span style="font-size:1.5em; font-weight:bold;">{pct}%</span>
      </div>
      <div style="width:100%; height:20px; background:rgba(255,255,255,0.2); border-radius:10px; overflow:hidden; margin-bottom:16px;">
        <div style="width:{pct}%; height:100%; background:rgba(255,255,255,0.9); transition: width 0.5s ease; border-radius:10px;"></div>
      </div>
      <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:12px; font-size:0.9em;">
        <div><b>{completed}</b><br>Completed</div>
        <div><b>{pending}</b><br>Pending</div>
        <div><b>{task_target}</b><br>Target</div>
      </div>
    </div>
    """

# --- Main Functions ---
def add_task(task_text, target_value):
    global weekly_tasks, task_target
    
    task_target = int(target_value) if target_value else 10
    task_text = task_text.strip()
    
    if not task_text:
        return [
            "⚠️ Please enter a task description.",
            "",
            format_list(),
            render_progress(),
            get_weekly_progress_gauge(),
            get_category_breakdown()
        ]
    
    if any(task_text.lower() == item["text"].lower() for item in weekly_tasks):
        return [
            "⚠️ This task already exists!",
            "",
            format_list(),
            render_progress(),
            get_weekly_progress_gauge(),
            get_category_breakdown()
        ]
    
    rephrased = hf_rephrase(task_text)
    weekly_tasks.append({
        "text": task_text,
        "added_at": time.strftime("%Y-%m-%d %H:%M"),
        "rephrased": rephrased
    })
    
    return [
        f"✅ Added: {task_text}",
        "",
        format_list(),
        render_progress(),
        get_weekly_progress_gauge(),
        get_category_breakdown()
    ]

def clear_all_tasks(target_value):
    global weekly_tasks, task_target
    
    task_target = int(target_value) if target_value else 10
    weekly_tasks.clear()
    
    return [
        "🗑️ All tasks cleared!",
        "",
        format_list(),
        render_progress(),
        get_weekly_progress_gauge(),
        get_category_breakdown()
    ]

def update_target(target_value):
    global task_target
    
    task_target = int(target_value) if target_value else 10
    
    return [
        f"🎯 Target updated to {task_target} tasks per week",
        gr.update(),
        format_list(),
        render_progress(),
        get_weekly_progress_gauge(),
        get_category_breakdown()
    ]

# --- UI ---
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray"),
    title="Weekly Tasks Dashboard"
) as demo:
    
    gr.Markdown("# 📊 Weekly Tasks Dashboard\n### Track your achievements, visualize progress, and stay motivated!")
    
    with gr.Row():
        with gr.Column(scale=4):
            task_in = gr.Textbox(
                label="💡 What did you achieve?",
                placeholder="e.g., Completed Q4 metrics dashboard for leadership review",
                lines=2
            )
        with gr.Column(scale=1):
            target_in = gr.Number(
                label="🎯 Weekly Target",
                value=10,
                precision=0
            )
    
    with gr.Row():
        add_btn = gr.Button("➕ Add Achievement", variant="primary", scale=2)
        clear_btn = gr.Button("🗑️ Clear All", variant="stop", scale=1)
        target_btn = gr.Button("⚙️ Update Target", scale=1)
    
    status = gr.Markdown("Ready to track your achievements! 🚀")
    
    progress_html = gr.HTML(render_progress())
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Progress Analytics")
            
            # Main gauge chart
            gauge_chart = gr.Plot(label="Weekly Progress Tracker")
            
            # Category breakdown
            category_chart = gr.Plot(label="Task Categories")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📝 This Week's Achievements")
            task_list = gr.Textbox(
                label="",
                interactive=False,
                lines=12,
                value=format_list()
            )
            
            email_btn = gr.Button("📧 Generate Professional Email", variant="secondary", size="sm")
            
            email_output = gr.Textbox(
                label="Professional Email Draft",
                interactive=True,
                lines=12,
                visible=False,
                show_copy_button=True
            )
    
    # Initialize charts on load
    demo.load(
        fn=lambda: [get_weekly_progress_gauge(), get_category_breakdown()],
        outputs=[gauge_chart, category_chart]
    )
    
    # Wire up events
    add_btn.click(
        fn=add_task,
        inputs=[task_in, target_in],
        outputs=[status, task_in, task_list, progress_html, gauge_chart, category_chart]
    )
    
    task_in.submit(
        fn=add_task,
        inputs=[task_in, target_in],
        outputs=[status, task_in, task_list, progress_html, gauge_chart, category_chart]
    )
    
    clear_btn.click(
        fn=clear_all_tasks,
        inputs=[target_in],
        outputs=[status, task_in, task_list, progress_html, gauge_chart, category_chart]
    )
    
    target_btn.click(
        fn=update_target,
        inputs=[target_in],
        outputs=[status, task_in, task_list, progress_html, gauge_chart, category_chart]
    )
    
    # Email generation
    email_btn.click(
        fn=lambda: [generate_professional_email(), gr.update(visible=True)],
        outputs=[email_output, email_output]
    )

if __name__ == "__main__":
    demo.launch(share=True, show_api=False)

#Done