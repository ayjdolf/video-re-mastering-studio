import streamlit as st
import streamlit.components.v1 as components
import time
import os
import pandas as pd
import whisper
import cv2
import numpy as np
import zipfile
import io
import sys
import subprocess
from moviepy.editor import VideoFileClip

# --- [특단의 조치] 라이브러리 자동 설치 & 업데이트 ---
# 프로그램 시작 시 자동으로 최신 버전을 설치합니다.
try:
    import google.generativeai as genai
    # 버전이 너무 낮으면 404 에러가 나므로 강제 업데이트 시도
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "google-generativeai"])
    import google.generativeai as genai
except ImportError:
    st.warning("⚠️ AI 부품이 없어서 설치 중입니다... 잠시만 기다려주세요!")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "google-generativeai"])
    import google.generativeai as genai
    st.success("✅ 설치 완료! 자동으로 다시 시작됩니다.")
    time.sleep(1)
    st.rerun()

# --- 기본 설정 ---
st.set_page_config(page_title="AI 영상 리마스터링 스튜디오", layout="wide")

if not os.path.exists("extracted_slides"):
    os.makedirs("extracted_slides")

# --- 세션 상태 ---
if 'script_df' not in st.session_state:
    st.session_state.script_df = None
if 'slides_data' not in st.session_state:
    st.session_state.slides_data = None
if 'storyboard_df' not in st.session_state:
    st.session_state.storyboard_df = None

# --- 스크롤 함수 ---
def scroll_to_bottom():
    js = """
    <script>
        var body = window.parent.document.body;
        setTimeout(function() {
            window.parent.scrollTo(0, body.scrollHeight);
        }, 500);
    </script>
    """
    components.html(js, height=0)

# --- 기능 함수들 ---
def extract_audio(video_path):
    audio_path = "temp_audio.mp3"
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='mp3', logger=None)
        return audio_path
    except Exception as e:
        return None

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base") 

def analyze_audio(audio_path, model):
    result = model.transcribe(audio_path)
    return result['segments']

def analyze_scenes(video_path, cut_x_ratio, cut_y_ratio, sensitivity, min_interval):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    saved_slides = []
    last_saved_frame = None 
    last_saved_time = -999 
    
    interval = int(fps) 
    progress_bar = st.progress(0)
    
    for i, frame_idx in enumerate(range(0, total_frames, interval)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        if i % 10 == 0:
            progress_bar.progress(frame_idx / total_frames)
            
        current_time = frame_idx / fps

        if (current_time - last_saved_time) < min_interval:
            continue

        h, w, _ = frame.shape
        
        # Masking
        analyze_frame = frame.copy()
        x_start = int(w * cut_x_ratio)
        y_start = int(h * cut_y_ratio)
        analyze_frame[y_start:h, x_start:w] = 0
        
        # Change Detection
        gray = cv2.cvtColor(analyze_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        is_new_slide = False
        
        if last_saved_frame is None:
            is_new_slide = True 
        else:
            score = cv2.absdiff(last_saved_frame, gray)
            score_mean = np.mean(score)
            
            if score_mean > sensitivity: 
                is_new_slide = True
        
        if is_new_slide:
            filename = f"extracted_slides/slide_{int(current_time)}.jpg"
            
            debug_frame = frame.copy()
            cv2.rectangle(debug_frame, (0, 0), (w, h), (0, 255, 0), 2)
            cv2.rectangle(debug_frame, (x_start, y_start), (w, h), (0, 0, 255), -1)
            cv2.putText(debug_frame, "IGNORED", (x_start + 10, y_start + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imwrite(filename, debug_frame) 
            
            saved_slides.append({
                "시간": time.strftime('%H:%M:%S', time.gmtime(current_time)),
                "초": current_time,
                "파일명": filename
            })
            
            last_saved_frame = gray 
            last_saved_time = current_time 
            
    cap.release()
    progress_bar.empty()
    return saved_slides

def create_slide_based_storyboard(script_df, slides):
    df_slides = pd.DataFrame(slides)
    df_slides = df_slides.sort_values(by="초")
    
    storyboard_data = []
    
    for i in range(len(df_slides)):
        current_slide = df_slides.iloc[i]
        start_time = current_slide['초']
        
        if i < len(df_slides) - 1:
            end_time = df_slides.iloc[i+1]['초']
        else:
            end_time = 999999 
            
        mask = (script_df['시작_초'] >= start_time) & (script_df['시작_초'] < end_time)
        matched_scripts = script_df[mask]
        
        full_text = " ".join(matched_scripts['내용'].tolist())
        
        storyboard_data.append({
            "No": i + 1, 
            "Time": f"{current_slide['시간']} ~ {time.strftime('%H:%M:%S', time.gmtime(end_time)) if end_time != 999999 else 'End'}",
            "Script": full_text,
            "Image": current_slide['파일명'],
            "AI_Description": "" 
        })
        
    return pd.DataFrame(storyboard_data)

def analyze_image_with_gemini(image_path, api_key):
    try:
        genai.configure(api_key=api_key)
        # 만약 이것도 안 되면 'gemini-1.5-pro' 로 변경 가능
        model = genai.GenerativeModel('gemini-1.5-flash') 
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(img)
        
        prompt = """
        이 이미지는 교육 영상의 한 장면(PPT 슬라이드)이야. 
        이 슬라이드를 나중에 AI 이미지 생성기로 다시 그릴 수 있도록 자세히 묘사해줘.
        다음 내용을 포함해서 한글로 3문장 이내로 요약해:
        1. 시각적 요소 (배경 스타일, 그림, 레이아웃)
        2. 주요 텍스트 내용이나 인용구 (OCR)
        3. 전체적인 분위기나 상황
        """
        
        response = model.generate_content([prompt, pil_img])
        return response.text
    except Exception as e:
        # 에러 메시지를 좀 더 자세히 출력
        return f"분석 실패: {str(e)}"

def create_zip_file(folder_path):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, arcname=file)
    return zip_buffer.getvalue()

# --- 메인 UI ---
st.title("🎥 AI Video Re-Mastering Studio")

with st.sidebar:
    st.header("1. 파일 입력")
    video_source = st.file_uploader("강의 영상 업로드", type=['mp4', 'avi', 'mov'])
    
    st.divider()
    st.header("⚙️ 설정 (Settings)")
    gemini_api_key = st.text_input("💎 Gemini API Key (선택)", type="password", help="키를 입력하면 AI가 이미지를 분석해줍니다.")
    
    st.divider()
    st.subheader("정밀 분석 설정")
    cut_x_input = st.slider("가로 위치", 0.5, 0.95, 0.75, 0.05)
    cut_y_input = st.slider("세로 위치", 0.3, 0.9, 0.6, 0.05)
    sensitivity_input = st.slider("민감도", 1.0, 20.0, 5.0)
    min_interval_input = st.slider("쿨타임", 1, 60, 5)

if video_source:
    with open("temp_video.mp4", "wb") as f:
        f.write(video_source.read())
    
    st.info("✅ 영상 준비 완료!")
    
    tab1, tab2 = st.tabs(["🔍 1단계: 재료 추출", "📝 2단계: 스토리보드"])
    
    # --- [탭 1] ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 (1) 음성 대본 추출", use_container_width=True):
                model = load_whisper_model()
                audio_file = extract_audio("temp_video.mp4")
                if audio_file:
                    with st.spinner("듣는 중..."):
                        segments = analyze_audio(audio_file, model)
                        data = [{"시작": time.strftime('%H:%M:%S', time.gmtime(s['start'])),
                                 "시작_초": s['start'], "내용": s['text']} for s in segments]
                        st.session_state.script_df = pd.DataFrame(data)
                        st.success("완료!")
                        scroll_to_bottom() 
            if st.session_state.script_df is not None:
                st.dataframe(st.session_state.script_df, height=300)

        with col2:
            if st.button("🎨 (2) PPT 장면 추출", use_container_width=True):
                with st.spinner("보는 중..."):
                    slides = analyze_scenes("temp_video.mp4", cut_x_input, cut_y_input, sensitivity_input, min_interval_input)
                    if slides:
                        st.session_state.slides_data = slides
                        st.success(f"{len(slides)}장 추출 완료!")
                        scroll_to_bottom() 
            
            if st.session_state.slides_data is not None:
                st.write(f"총 {len(st.session_state.slides_data)}장의 PPT 확보")
                
                zip_data = create_zip_file("extracted_slides")
                st.download_button("📦 모든 이미지 다운로드 (.ZIP)", zip_data, "ppt_slides.zip", "application/zip", type="primary")
                
                with st.expander("📸 전체 장면 펼쳐보기"):
                    cols = st.columns(3)
                    for idx, slide in enumerate(st.session_state.slides_data):
                        with cols[idx % 3]:
                            st.image(slide['파일명'], caption=f"Scene #{idx+1} [{slide['시간']}]", use_container_width=True)

    # --- [탭 2] ---
    with tab2:
        st.subheader("📝 장면(Scene) 리스트 & AI 분석")
        
        if st.session_state.script_df is None or st.session_state.slides_data is None:
            st.warning("⚠️ 1단계에서 음성과 이미지를 모두 추출해주세요.")
        else:
            if st.session_state.storyboard_df is None:
                st.session_state.storyboard_df = create_slide_based_storyboard(st.session_state.script_df, st.session_state.slides_data)
            
            c1, c2 = st.columns([1, 1])
            with c1:
                if gemini_api_key:
                    if st.button("🤖 AI 장면 정밀 분석 시작 (Gemini)", type="primary"):
                        progress_bar = st.progress(0)
                        total = len(st.session_state.storyboard_df)
                        for index, row in st.session_state.storyboard_df.iterrows():
                            if not row['AI_Description']:
                                desc = analyze_image_with_gemini(row['Image'], gemini_api_key)
                                st.session_state.storyboard_df.at[index, 'AI_Description'] = desc
                            progress_bar.progress((index + 1) / total)
                        
                        st.success("분석 완료!")
                        scroll_to_bottom() 
                        st.rerun()
                else:
                    st.info("💡 사이드바에 Gemini API 키를 넣으면 이미지 분석이 가능합니다.")

            with c2:
                csv_sb = st.session_state.storyboard_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("💾 전체 리스트 다운로드 (Excel)", csv_sb, 'storyboard_final.csv', 'text/csv', type="primary")
            
            st.divider()
            
            # [5단 리스트 출력] 
            for index, row in st.session_state.storyboard_df.iterrows():
                cols = st.columns([0.4, 0.8, 2.5, 1.5, 1.5])
                
                with cols[0]:
                    st.markdown(f"**#{row['No']}**")
                
                with cols[1]:
                    st.caption(row['Time'])
                
                with cols[2]:
                    st.text_area(f"s_{index}", row['Script'], height=120, label_visibility="collapsed")
                    
                with cols[3]:
                    st.image(row['Image'], use_container_width=True)
                    
                with cols[4]:
                    if row['AI_Description']:
                        if "분석 실패" in row['AI_Description']:
                             st.error("Error: 키 확인 필요")
                             with st.expander("에러 내용 보기"):
                                 st.write(row['AI_Description'])
                        else:
                            st.info(row['AI_Description'])
                    else:
                        st.caption("Waiting...")
                
                st.markdown("---")