# 🌐 Epic/Story Generator - Web Interface

A beautiful web application for your AI model that transforms project descriptions into structured epics, user stories, and story points.

---

## ✨ Features

- **Modern Web UI** - Clean, responsive design that works on all devices
- **Real-time Generation** - Get results in 1-2 seconds
- **Example Templates** - Quick-start with pre-built examples
- **GPU Accelerated** - Uses your Quadro RTX 3000 for fast inference
- **Copy & Paste Ready** - Easy to use generated outputs in your PM tools

---

## 🚀 Quick Start

### Method 1: Double-click (Easiest)

1. Double-click **`START_WEBAPP.bat`**
2. Wait for "Server running at: http://localhost:5000"
3. Open your browser to **http://localhost:5000**
4. Done! 🎉

### Method 2: Command Line

```bash
py -3.12 web_app.py
```

Then open: **http://localhost:5000**

---

## 📖 How to Use

1. **Enter Description**
   - Type or paste your project description in the text area
   - Be specific for best results!

2. **Click "Generate"**
   - The AI will process your description
   - Results appear in ~1-2 seconds

3. **Review Results**
   - Epic Category
   - User Story
   - Story Points
   - Tasks (if generated)
   - Acceptance Criteria (if generated)

4. **Try Examples**
   - Click any example chip to auto-fill
   - Great for learning what works best

---

## 🎨 Screenshot Preview

```
┌─────────────────────────────────────────────────┐
│  🚀 Epic/Story Generator                        │
│  AI-Powered Project Planning                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  Enter Your Project Description                 │
│  ┌─────────────────────────────────────────┐   │
│  │ Build a mobile app for tracking...      │   │
│  │                                         │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  [✨ Generate Epic & User Story]  [🗑️ Clear]  │
│                                                 │
│  Quick Examples: [Chat App] [E-commerce]...    │
│                                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  📝 Your Input                                  │
│  Build a mobile app for tracking fitness...    │
│                                                 │
│  🎯 Epic Category                               │
│  Mobile                                         │
│                                                 │
│  👤 User Story                                  │
│  As a user, I want to...                       │
│                                                 │
│  📊 Story Points                                │
│  [2 Points]                                     │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 🔧 Technical Details

### Architecture

```
┌──────────────┐      HTTP      ┌──────────────┐
│   Browser    │ ←────────────→ │ Flask Server │
│  (Frontend)  │   JSON API     │  (Backend)   │
└──────────────┘                └──────┬───────┘
                                       │
                                       ↓
                              ┌─────────────────┐
                              │  T5 AI Model    │
                              │  (60.5M params) │
                              │  GPU Accelerated│
                              └─────────────────┘
```

### API Endpoints

#### `POST /api/generate`
Generate epic/story from description

**Request:**
```json
{
  "description": "Your project description"
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "epic": "Category",
    "user_story": "As a user, I want to...",
    "story_points": "2",
    "tasks": ["Task 1", "Task 2"],
    "acceptance_criteria": ["Criteria 1"]
  },
  "raw_output": "..."
}
```

#### `GET /api/examples`
Get example project descriptions

#### `GET /api/health`
Check server health status

---

## 📁 Files

- **`web_app.py`** - Flask backend server
- **`templates/index.html`** - Frontend UI
- **`START_WEBAPP.bat`** - Quick launch script
- **`requirements_webapp.txt`** - Web dependencies

---

## 🎯 Tips for Best Results

### ✅ Good Descriptions

```
✓ "Build a mobile app for iOS with user authentication, profile management, and real-time chat"
✓ "Create a REST API for customer data with CRUD operations and role-based permissions"
✓ "Develop a dashboard showing sales analytics by region with charts and export to PDF"
```

### ❌ Vague Descriptions

```
✗ "Make an app"
✗ "Build something"
✗ "Create feature"
```

### 💡 Pro Tips

1. **Be specific** - Include platform, features, and requirements
2. **Use natural language** - Write like talking to a PM
3. **Include context** - Mention integrations, users, or business goals
4. **Try examples** - Learn from the provided templates

---

## 🔌 Integration Ideas

### Export to Jira
Use the generated JSON to auto-create Jira issues:

```python
import requests

# Get generated result from web app
result = {
    "epic": "Mobile",
    "user_story": "As a user, I want to...",
    "story_points": "2"
}

# Create Jira issue
jira_api = "https://your-domain.atlassian.net/rest/api/3/issue"
payload = {
    "fields": {
        "project": {"key": "PROJ"},
        "summary": result["user_story"],
        "description": result["user_story"],
        "issuetype": {"name": "Story"},
        "customfield_10016": int(result["story_points"])  # Story points field
    }
}
```

### Use with GitHub Issues

```python
# Create GitHub issue from result
gh_api = "https://api.github.com/repos/owner/repo/issues"
payload = {
    "title": result["user_story"],
    "body": f"**Epic:** {result['epic']}\n\n**Story Points:** {result['story_points']}\n\n**Tasks:**\n" +
            "\n".join(f"- [ ] {task}" for task in result["tasks"]),
    "labels": [result["epic"]]
}
```

---

## 🛠️ Troubleshooting

### Server won't start

**Problem:** Port 5000 already in use

**Solution:**
```python
# Edit web_app.py, line 94, change port:
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Model loading fails

**Problem:** Can't find model file

**Solution:**
- Check that `models/epic-story-model/final/` exists
- Re-run training if model is missing

### Slow generation

**Problem:** Takes >5 seconds

**Solution:**
- Check GPU is being used (should show "Using device: cuda")
- Close other GPU applications
- First generation is slower (model loading)

### Browser can't connect

**Problem:** Connection refused

**Solution:**
- Make sure server is running (check terminal)
- Try http://127.0.0.1:5000 instead of localhost
- Check firewall settings

---

## 🚦 Production Deployment

### For Team Use

1. **Change host** in `web_app.py`:
   ```python
   app.run(debug=False, host='0.0.0.0', port=5000)
   ```

2. **Use production server** (Gunicorn):
   ```bash
   pip install gunicorn
   gunicorn -w 1 -b 0.0.0.0:5000 web_app:app
   ```

3. **Add authentication** (optional):
   - Add login page
   - Use Flask-Login
   - Implement API keys

4. **Use reverse proxy** (nginx):
   ```nginx
   location / {
       proxy_pass http://localhost:5000;
   }
   ```

### For Public Hosting

Use platforms like:
- **Heroku** - Easy deployment
- **Google Cloud Run** - Serverless containers
- **AWS EC2** - Full control with GPU support
- **DigitalOcean** - Simple VPS

**Note:** Make sure to include GPU support for best performance!

---

## 📊 Performance

- **Model Load Time**: ~2 seconds (one-time on startup)
- **Generation Time**: 1-2 seconds per description
- **Concurrent Users**: 1-5 (single GPU)
- **Memory Usage**: ~3.5GB GPU RAM
- **CPU Usage**: Minimal (GPU does the work)

---

## 🎉 You're All Set!

Your web interface is ready to use. Just run `START_WEBAPP.bat` and start generating epics and user stories!

**Questions or issues?** Check the troubleshooting section above.

**Want to customize?** Edit `templates/index.html` for UI changes or `web_app.py` for backend logic.

**Happy generating!** 🚀
