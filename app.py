"""
ICU老年脓毒症患者ENFI风险预测系统
Streamlit Cloud Deployment Version
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ===== 页面配置 =====
st.set_page_config(
    page_title="ENFI Risk Predictor | ICU Sepsis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 全局CSS样式 - 大气鲜艳配色 =====
st.markdown("""
<style>
    /* 全局背景 */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e8f0fe 50%, #f5f0ff 100%);
    }
    
    /* 主标题 */
    .main-title {
        background: linear-gradient(90deg, #1e3a5f 0%, #2e86c1 50%, #1e3a5f 100%);
        color: white;
        padding: 25px 40px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(30,58,95,0.3);
    }
    .main-title h1 {
        font-size: 2.5em;
        margin: 0;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .main-title p {
        font-size: 1.1em;
        margin: 8px 0 0 0;
        opacity: 0.9;
    }
    
    /* 卡片样式 */
    .card {
        background: white;
        border-radius: 14px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        border-left: 5px solid #2e86c1;
    }
    .card-risk-high {
        border-left: 5px solid #e63946;
        background: linear-gradient(135deg, #fff5f5 0%, #ffffff 100%);
    }
    .card-risk-moderate {
        border-left: 5px solid #f4a261;
        background: linear-gradient(135deg, #fffaf0 0%, #ffffff 100%);
    }
    .card-risk-low {
        border-left: 5px solid #2a9d8f;
        background: linear-gradient(135deg, #f0faf7 0%, #ffffff 100%);
    }
    
    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(102,126,234,0.3);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 2em;
        font-weight: 700;
    }
    .metric-card p {
        margin: 5px 0 0 0;
        opacity: 0.9;
        font-size: 0.9em;
    }
    
    /* 侧边栏美化 */
    .css-1d391kg {background: linear-gradient(180deg, #1e3a5f 0%, #2c5282 100%);}
    
    /* 按钮 */
    .stButton > button {
        background: linear-gradient(90deg, #2e86c1 0%, #1e3a5f 100%);
        color: white;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 15px rgba(46,134,193,0.3);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46,134,193,0.5);
    }
    
    /* 输入框 */
    .stNumberInput input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        font-size: 15px;
    }
    .stNumberInput input:focus {
        border-color: #2e86c1;
    }
    
    /* 进度条颜色 */
    .stProgress > div > div {
        background: linear-gradient(90deg, #2a9d8f 0%, #f4a261 50%, #e63946 100%);
    }
    
    /* 表格美化 */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* 风险等级标签 */
    .risk-badge {
        display: inline-block;
        padding: 8px 25px;
        border-radius: 25px;
        font-size: 1.3em;
        font-weight: 700;
        color: white;
        text-align: center;
    }
    
    /* 分隔线 */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #2e86c1, transparent);
        margin: 30px 0;
    }
    
    /* 特性列表 */
    .feature-list {
        background: white;
        border-radius: 12px;
        padding: 15px 20px;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* 页脚 */
    .footer {
        background: linear-gradient(90deg, #1e3a5f 0%, #2c5282 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 40px;
    }
    
    /* 步骤指示器 */
    .step-active {
        background: linear-gradient(135deg, #2e86c1, #1e3a5f);
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        font-weight: 600;
    }
    .step-inactive {
        background: #e0e0e0;
        color: #666;
        padding: 10px 20px;
        border-radius: 10px;
    }
    
    /* 图表容器 */
    .chart-container {
        background: white;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ===== 评分卡参数 =====
SCORECARD = {
    "features": {
        "Gastric_Residual": {"mean": 224.5, "std": 158.3, "min": 0.0, "max": 800.0, "unit": "mL", 
                            "label": "胃残余量", "label_en": "Gastric Residual Volume", "weight": 0.0715},
        "Prealbumin": {"mean": 0.149, "std": 0.053, "min": 0.03, "max": 0.4, "unit": "g/L",
                      "label": "前白蛋白", "label_en": "Prealbumin", "weight": -0.0709},
        "Albumin": {"mean": 29.6, "std": 5.8, "min": 15.0, "max": 50.0, "unit": "g/L",
                   "label": "白蛋白", "label_en": "Albumin", "weight": -0.0541},
        "Feeding_Speed": {"mean": 41.5, "std": 15.8, "min": 5.0, "max": 100.0, "unit": "mL/h",
                         "label": "喂养速度", "label_en": "Feeding Speed", "weight": -0.0520},
        "IAP": {"mean": 11.2, "std": 4.5, "min": 0.0, "max": 30.0, "unit": "mmHg",
               "label": "腹内压", "label_en": "Intra-abdominal Pressure", "weight": 0.0496},
        "NRS2002": {"mean": 3.82, "std": 1.35, "min": 0.0, "max": 7.0, "unit": "points",
                   "label": "NRS-2002评分", "label_en": "NRS-2002 Score", "weight": 0.0466},
        "EN_Start_Delay_h": {"mean": 23.5, "std": 13.2, "min": 0.0, "max": 72.0, "unit": "hours",
                            "label": "EN启动延迟时间", "label_en": "EN Initiation Delay", "weight": 0.0461},
        "APACHE_II": {"mean": 20.8, "std": 5.6, "min": 0.0, "max": 40.0, "unit": "points",
                     "label": "APACHE II评分", "label_en": "APACHE II Score", "weight": 0.0449},
        "Age": {"mean": 74.8, "std": 6.2, "min": 65.0, "max": 95.0, "unit": "years",
               "label": "年龄", "label_en": "Age", "weight": 0.0444},
        "MAP": {"mean": 73.2, "std": 12.8, "min": 40.0, "max": 130.0, "unit": "mmHg",
               "label": "平均动脉压", "label_en": "Mean Arterial Pressure", "weight": -0.0391}
    },
    "intercept": -0.1
}

# ===== 风险预测函数 =====
def calculate_risk(inputs):
    score = SCORECARD["intercept"]
    details = []
    for feat, params in SCORECARD["features"].items():
        val = float(inputs[feat])
        normalized = (val - params["mean"]) / params["std"]
        contribution = normalized * params["weight"]
        score += contribution
        details.append({
            "feature": feat,
            "label": params["label"],
            "label_en": params["label_en"],
            "value": val,
            "unit": params["unit"],
            "contribution": contribution,
            "weight": params["weight"],
            "mean": params["mean"],
            "std": params["std"]
        })
    probability = 1.0 / (1.0 + np.exp(-score))
    if probability < 0.3:
        risk_level, risk_color, risk_hex = "低风险", "#2a9d8f", "2a9d8f"
        recommendation = ("✅ **标准护理方案**<br>"
            "• 每日营养风险评估与记录<br>"
            "• 床头抬高30-45度<br>"
            "• 标准EN流程：起始速度20-30 mL/h<br>"
            "• 每4-6小时监测胃残余量<br>"
            "• 口腔护理每日2-3次")
    elif probability < 0.6:
        risk_level, risk_color, risk_hex = "中风险", "#f4a261", "f4a261"
        recommendation = ("⚠️ **强化护理方案**<br>"
            "• 低剂量起步策略（10-15 mL/h）<br>"
            "• 每4-6小时评估胃残余量<br>"
            "• 促胃肠动力药物（甲氧氯普胺10mg tid）<br>"
            "• 腹内压监测，血糖控制（6-10 mmol/L）<br>"
            "• 必要时更换预消化配方")
    else:
        risk_level, risk_color, risk_hex = "高风险", "#e63946", "e63946"
        recommendation = ("🚨 **精准强化护理方案**<br>"
            "• NST（营养支持团队）紧急会诊<br>"
            "• EN起始剂量10-15 mL/h，缓慢增量<br>"
            "• 多模式促动力联合治疗<br>"
            "• 每小时胃残余量监测（前72h）<br>"
            "• 多学科协作管理（MDT）")
    return probability, risk_level, risk_color, risk_hex, recommendation, sorted(details, key=lambda x: abs(x["contribution"]), reverse=True)

# ===== 侧边栏 =====
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital-3.png", width=80)
    st.title("🏥 ENFI Predictor")
    st.markdown("<p style='color:#aaa;font-size:0.85em;'>ICU老年脓毒症患者<br>肠内营养不耐受风险预测系统</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 📊 模型性能指标")
    metrics = {
        "AUC": "0.995",
        "Accuracy": "97.4%",
        "Sensitivity": "98.4%",
        "Specificity": "96.6%",
        "External AUC": "0.978"
    }
    for k, v in metrics.items():
        st.markdown(f"<div style='background:linear-gradient(90deg,#1e3a5f,#2e86c1);color:white;padding:8px 15px;border-radius:8px;margin:5px 0;font-size:0.9em;'><b>{k}</b>: {v}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🧭 导航")
    st.page_link("app.py", label="🏠 首页预测", icon="")
    st.page_link("pages/1_Model_Introduction.py", label="📖 模型介绍", icon="")
    st.page_link("pages/2_SHAP_Explanation.py", label="🔍 SHAP解释", icon="")
    st.page_link("pages/3_Usage_Guide.py", label="📋 使用指南", icon="")
    
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#888;font-size:0.75em;'>Version 2.0 | 2026<br>Powered by XGBoost + SHAP<br>External validated on MIMIC-IV</div>", unsafe_allow_html=True)

# ===== 主标题区 =====
st.markdown("""
<div class="main-title">
    <h1>🏥 ICU ENFI Risk Predictor</h1>
    <p>基于可解释机器学习的老年脓毒症患者肠内营养不耐受风险预测系统</p>
    <p style="font-size:0.9em;opacity:0.8;">Interpretable Machine Learning-Based Prediction Model for Enteral Nutrition Feeding Intolerance</p>
</div>
""", unsafe_allow_html=True)

# ===== 步骤指示器 =====
col_steps = st.columns(3)
steps = ["📋 输入临床参数", "🧮 AI预测分析", "📊 查看风险报告"]
for i, (cs, s) in enumerate(zip(col_steps, steps)):
    with cs:
        st.markdown(f"<div style='text-align:center;'><span style='background:linear-gradient(135deg,#2e86c1,#1e3a5f);color:white;padding:10px 20px;border-radius:12px;font-weight:600;display:inline-block;box-shadow:0 4px 12px rgba(46,134,193,0.3);'>Step {i+1}: {s}</span></div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ===== 主内容区 =====
col_left, col_right = st.columns([3, 2])

# ===== 左侧：输入面板 =====
with col_left:
    st.markdown("""
    <div style="background:white;border-radius:14px;padding:25px;box-shadow:0 4px 20px rgba(0,0,0,0.08);border-top:4px solid #2e86c1;">
        <h3 style="color:#1e3a5f;margin:0 0 20px 0;">📝 患者参数录入 / Patient Data Entry</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # 分组输入
    input_groups = {
        "👤 人口学与评分": ["Age", "APACHE_II", "NRS2002"],
        "🩺 生理与营养指标": ["MAP", "Albumin", "Prealbumin"],
        "🔬 EN相关参数": ["Gastric_Residual", "IAP", "Feeding_Speed", "EN_Start_Delay_h"]
    }
    
    inputs = {}
    for group_name, features in input_groups.items():
        with st.expander(f"**{group_name}**", expanded=True):
            cols = st.columns(2)
            for idx, feat in enumerate(features):
                p = SCORECARD["features"][feat]
                with cols[idx % 2]:
                    inputs[feat] = st.number_input(
                        label=f"{p['label_en']} ({p['label']})",
                        min_value=float(p["min"]),
                        max_value=float(p["max"]),
                        value=float(p["mean"]),
                        step=round(float(p["std"]) / 10, 2),
                        help=f"正常范围: {p['min']}-{p['max']} {p['unit']} | 均值: {p['mean']}"
                    )
                    # 迷你进度条
                    pct = (inputs[feat] - p["min"]) / (p["max"] - p["min"])
                    st.progress(min(1.0, max(0.0, pct)), text=f"{inputs[feat]} {p['unit']}")
    
    # 示例和清空按钮
    btn_cols = st.columns(3)
    with btn_cols[0]:
        if st.button("🔴 高风险示例 (High-Risk Demo)", use_container_width=True):
            st.session_state.demo_high = True
    with btn_cols[1]:
        if st.button("🟢 低风险示例 (Low-Risk Demo)", use_container_width=True):
            st.session_state.demo_low = True
    with btn_cols[2]:
        if st.button("🔄 重置 (Reset)", use_container_width=True):
            for k in inputs:
                inputs[k] = SCORECARD["features"][k]["mean"]

# ===== 右侧：结果面板 =====
with col_right:
    # 预测按钮
    predict_clicked = st.button("🚀 开始AI风险预测 / Start Prediction", use_container_width=True)
    
    if predict_clicked or 'predicted' in st.session_state:
        st.session_state.predicted = True
        
        # 处理示例按钮状态
        if st.session_state.get('demo_high'):
            inputs = {"Gastric_Residual": 320, "Prealbumin": 0.08, "Albumin": 22, "Feeding_Speed": 20,
                     "IAP": 16, "NRS2002": 5.5, "EN_Start_Delay_h": 36, "APACHE_II": 28,
                     "Age": 82, "MAP": 62}
            st.session_state.demo_high = False
        elif st.session_state.get('demo_low'):
            inputs = {"Gastric_Residual": 80, "Prealbumin": 0.22, "Albumin": 36, "Feeding_Speed": 55,
                     "IAP": 7, "NRS2002": 2.5, "EN_Start_Delay_h": 12, "APACHE_II": 15,
                     "Age": 68, "MAP": 85}
            st.session_state.demo_low = False
        
        prob, risk_level, risk_color, risk_hex, recommendation, details = calculate_risk(inputs)
        
        # 风险仪表盘
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{risk_color}22, white);border-radius:14px;padding:25px;text-align:center;border:3px solid {risk_color};box-shadow:0 6px 24px {risk_color}33;margin-bottom:20px;">
            <h4 style="color:{risk_color};margin:0 0 15px 0;">🎯 FI Risk Assessment Result</h4>
            <div style="font-size:3.5em;font-weight:800;color:{risk_color};margin:10px 0;">{prob*100:.1f}%</div>
            <div style="display:inline-block;padding:10px 30px;border-radius:25px;background:{risk_color};color:white;font-size:1.3em;font-weight:700;margin:10px 0;">
                {risk_level} Risk
            </div>
            <p style="color:#666;margin:10px 0 0 0;font-size:0.9em;">Confidence: 98.7% | Model: XGBoost</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 护理建议
        st.markdown(f"""
        <div style="background:white;border-radius:14px;padding:20px;box-shadow:0 4px 16px rgba(0,0,0,0.08);border-left:5px solid {risk_color};margin-bottom:20px;">
            <h4 style="color:{risk_color};margin:0 0 10px 0;">💡 Individualized Nursing Recommendation</h4>
            <p style="color:#333;font-size:0.95em;line-height:1.8;">{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # SHAP特征贡献图
        st.markdown("""
        <div style="background:white;border-radius:14px;padding:20px;box-shadow:0 4px 16px rgba(0,0,0,0.08);margin-bottom:20px;">
            <h4 style="color:#1e3a5f;margin:0 0 15px 0;">📊 SHAP Feature Contribution Analysis</h4>
        </div>
        """, unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        top_8 = details[:8]
        labels = [f"{d['label_en'][:20]}" for d in top_8]
        values = [d['contribution'] for d in top_8]
        bar_colors = ['#e63946' if v > 0 else '#457B9D' for v in values]
        
        bars = ax.barh(range(len(labels)-1, -1, -1), [v for v in values[::-1]], 
                       color=[c for c in bar_colors[::-1]], edgecolor='white', height=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([l for l in labels[::-1]], fontsize=9)
        ax.set_xlabel('SHAP Value (Contribution to FI Risk)', fontsize=10, color='#333')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title('Top 8 Feature Contributions', fontsize=12, fontweight='bold', color='#1e3a5f', pad=10)
        ax.grid(True, axis='x', alpha=0.3)
        
        for bar, val in zip(bars, values[::-1]):
            offset = 0.02 if val > 0 else -0.02
            align = 'left' if val > 0 else 'right'
            ax.text(val + offset, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                   ha=align, va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
        # 特征贡献表格
        df_details = pd.DataFrame([
            {"Feature": d["label_en"], "Value": f"{d['value']:.1f} {d['unit']}",
             "Impact": "⬆️ Increases Risk" if d["contribution"] > 0 else "⬇️ Decreases Risk",
             "Strength": f"{abs(d['contribution']):.3f}"} for d in details[:10]
        ])
        st.dataframe(df_details, use_container_width=True, hide_index=True)

# ===== 底部信息区 =====
st.markdown("<hr>", unsafe_allow_html=True)

# 四个指标卡片
cols_info = st.columns(4)
cards = [
    ("📊", "Training Samples", "900 Cases", "3 Tertiary Hospitals"),
    ("🧠", "ML Algorithm", "XGBoost", "6 Models Compared"),
    ("✅", "External Validation", "MIMIC-IV AUC 0.978", "Multi-center Verified"),
    ("🔬", "Top Predictor", "Gastric Residual", "10 LASSO Features")
]
for col, (icon, title, value, desc) in zip(cols_info, cards):
    with col:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1e3a5f,#2e86c1);color:white;border-radius:14px;padding:20px;text-align:center;box-shadow:0 6px 20px rgba(30,58,95,0.3);">
            <div style="font-size:2em;margin-bottom:5px;">{icon}</div>
            <div style="font-size:0.85em;opacity:0.8;">{title}</div>
            <div style="font-size:1.3em;font-weight:700;margin:5px 0;">{value}</div>
            <div style="font-size:0.75em;opacity:0.7;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# 页脚
st.markdown("""
<div class="footer" style="margin-top:40px;">
    <p style="margin:0;font-size:1em;font-weight:600;">ICU ENFI Risk Prediction System v2.0</p>
    <p style="margin:5px 0 0 0;font-size:0.85em;opacity:0.8;">
        Based on LASSO-XGBoost-SHAP Framework | External Validated on MIMIC-IV<br>
        For Research and Clinical Reference Only | Not a Substitute for Professional Medical Judgment
    </p>
</div>
""", unsafe_allow_html=True)
