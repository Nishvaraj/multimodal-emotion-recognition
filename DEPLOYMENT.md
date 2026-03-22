# MMER Deployment Guide - Railway + Vercel

## Overview
- **Backend**: FastAPI on Railway
- **Frontend**: React on Vercel
- **Database**: Supabase (PostgreSQL)
- **Auth**: Supabase Authentication

---

## ✅ Pre-Deployment Checklist

### 1. **Supabase Database Setup** (5 min) ⚠️ DO THIS FIRST

**You need to:**
1. Open Supabase Dashboard → Your Project → **SQL Editor**
2. Click **New Query**
3. Copy-paste contents of `supabase_schema.sql` from your project root
4. Click **Run** ▶️
5. Verify success (should show `"Success. No rows returned"`)

**What this creates:**
- ✅ `analysis_history` table with RLS policies
- ✅ Users can only see their own records
- ✅ Automatic timestamps for created_at/updated_at
- ✅ Indexes for performance

---

### 2. **Backend Configuration - Railway**

**Files created:**
- ✅ `Procfile` - How Railway starts your app
- ✅ `backend/config.py` - Environment-based config
- ✅ `requirements.txt` - Updated with gunicorn
- ✅ `backend/main.py` - Updated with logging + CORS

**Environment Variables (Railway Dashboard):**
```
ENV=production
USE_GPU=false
```

---

### 3. **Frontend Configuration - Vercel**

**Files updated:**
- ✅ `.env.local` - Local dev settings
- ✅ `frontend/src/App.js` - API_BASE from env var
- ✅ `frontend/src/supabaseHistoryService.js` - Supabase DB service

**Environment Variables (Vercel Dashboard):**
```
REACT_APP_SUPABASE_URL=https://anrbndmgaofenifhzrpg.supabase.co
REACT_APP_SUPABASE_ANON_KEY=sb_publishable_NrM26NwAu3T_wdFopmLz2A_lP7w8epB
REACT_APP_API_BASE=https://your-railway-backend.up.railway.app
```

---

## 🚀 Step-by-Step Deployment

### **Step 1: Setup Supabase DB (5 min) — DO FIRST**

1. Go to https://app.supabase.com/project/YOUR_PROJECT/sql
2. Click **New Query**
3. Paste entire `supabase_schema.sql` file
4. Click **Run**
5. ✅ You should see success message

---

### **Step 2: Deploy Backend to Railway (5 min)**

1. Go to https://railway.app and sign in
2. Click **New Project** → **Deploy from GitHub**
3. Select your repository
4. Wait for Railway to auto-detect Python project
5. Go to **Variables** and add:
   ```
   ENV=production
   USE_GPU=false
   ```
6. Railway auto-deploys on push to `main`
7. Get your Railway URL from **Deployments** → Copy the URL (e.g., `https://your-project-production.up.railway.app`)

**What happens:**
- First deployment: ~2-3 min (quick)
- First API call: ~8-10 min (downloading ML models from Hugging Face)
- Subsequent API calls: ~2-5 sec

---

### **Step 3: Update Frontend Environment Variables**

Once you have your Railway backend URL:

1. Create `.env.local` in `frontend/` folder:
   ```
   REACT_APP_SUPABASE_URL=https://anrbndmgaofenifhzrpg.supabase.co
   REACT_APP_SUPABASE_ANON_KEY=sb_publishable_NrM26NwAu3T_wdFopmLz2A_lP7w8epB
   REACT_APP_API_BASE=https://your-railway-backend.up.railway.app
   ```

2. Test locally:
   ```bash
   npm --prefix frontend start
   # Visit http://localhost:3001 and test signup/analysis/history
   ```

---

### **Step 4: Deploy Frontend to Vercel (5 min)**

1. Go to https://vercel.com and sign in with GitHub
2. Click **Add New** → **Project**
3. Select your GitHub repository
4. Click **Import**
5. Go to **Environment Variables** and add:
   ```
   REACT_APP_SUPABASE_URL=https://anrbndmgaofenifhzrpg.supabase.co
   REACT_APP_SUPABASE_ANON_KEY=sb_publishable_NrM26NwAu3T_wdFopmLz2A_lP7w8epB
   REACT_APP_API_BASE=https://your-railway-backend.up.railway.app
   ```
6. Click **Deploy**
7. Wait for deployment to complete (~2 min)
8. Get your Vercel URL from the deployment page

---

### **Step 5: Update Backend CORS (Optional but Recommended)**

Once you have your Vercel URL (e.g., `https://mmer.vercel.app`):

1. Edit `backend/config.py`:
   ```python
   allowed_origins = [
       "https://mmer.vercel.app",  # Your Vercel URL
       "https://yourfrontend.vercel.app"
   ]
   ```

2. Commit and push to GitHub
3. Railway auto-redeploys with updated CORS

---

## ✅ Testing Checklist

After deployment, test these flows:

### **Test Local Dev First:**
```bash
# Terminal 1: Backend
cd backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2: Frontend
npm --prefix frontend start
# Visit http://localhost:3001
```

**Test cases:**
- [ ] Signup with email/password
- [ ] Login with credentials
- [ ] Run facial emotion analysis
- [ ] Run speech emotion analysis
- [ ] Check history in Supabase (History tab)
- [ ] Pin/unpin record
- [ ] Add note to record
- [ ] Export CSV
- [ ] Export Summary Report
- [ ] Logout and login again (history persists)

---

### **Test Production Deployment:**
```
https://your-frontend.vercel.app
```

**Test cases:**
- [ ] Signup → redirects to dashboard
- [ ] Facial analysis → history appears
- [ ] Speech analysis → history appears
- [ ] Refresh page → history persists
- [ ] Open in different browser → same history (Supabase sync!)
- [ ] Export works

---

## 📊 Performance Expectations

| Operation | Time | Notes |
|---|---|---|
| **First API call after deploy** | 8-10 min | Models downloading from HuggingFace |
| **Facial prediction** | 2-3 sec | CPU-dependent |
| **Speech prediction** | 2-5 sec | CPU-dependent |
| **Load history** | <100ms | Supabase query + filtering |
| **Save analysis** | <500ms | Network + Supabase write |

---

## 🔧 Troubleshooting

### Frontend Can't Connect to Backend
- Check `REACT_APP_API_BASE` env var in Vercel
- Ensure Railway backend is running (check Deployments page)
- Open browser console (`F12`) and check Network tab

### Supabase History Not Appearing
- Verify SQL schema was run (check Supabase Tables)
- Check browser console for errors
- Verify user is authenticated

### Models Loading Forever (First Boot)
- Normal! First API call downloads ~2GB models
- Takes 8-10 minutes on first deployment
- Subsequent calls are fast (2-5 sec)

---

## 📝 Environment Variables Summary

### Backend (Railway dashboard)
```
ENV=production
USE_GPU=false
REACT_APP_VERCEL_URL=https://your-frontend.vercel.app
```

### Frontend (Vercel dashboard)
```
REACT_APP_SUPABASE_URL=https://anrbndmgaofenifhzrpg.supabase.co
REACT_APP_SUPABASE_ANON_KEY=sb_publishable_NrM26NwAu3T_wdFopmLz2A_lP7w8epB
REACT_APP_API_BASE=https://your-railway-backend.up.railway.app
```

---

## 🎉 Success Criteria

Your deployment is successful when:

✅ Frontend loads at Vercel URL
✅ Can signup and login
✅ Can run facial/speech/multimodal analysis
✅ History appears and persists across login/logout
✅ Can pin/delete/edit notes on history items
✅ Can export CSV and summary reports

---

## 🚀 Next Steps (Optional)

- [ ] Set up automatic email verification in Supabase
- [ ] Add error tracking (Sentry)
- [ ] Set up monitoring/alerts (Railway dashboard)
- [ ] Optimize model loading (S3 caching)
- [ ] Add analytics

---

**Ready to deploy?** Follow the steps above in order! Good luck! 🚀
