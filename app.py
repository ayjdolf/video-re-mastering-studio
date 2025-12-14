import streamlit as st
import cv2
import os
import shutil
import numpy as np
import pandas as pd
import google.generativeai as genai
from moviepy.editor import VideoFileClip

# ==========================================
# 1. 환경 설정
# ==========================================
BASE_DIR = os.getcwd()
TEMP_DIR = os.path.join(BASE_DIR, "temp_workspace")
OUTPUT_DIR = os.path.join(BASE_DIR, "extracted_scenes")
PPT_DIR = os.path.join(BASE_DIR, "uploaded_ppts")
AUDIO_PATH = os.path.join(TEMP_DIR, "audio.mp3")

# ==========================================
# 2. 핵심 분석 엔진 (Track 1 & Track 2)
# ==========================================

# [Track 1] PPT 원본과 비교해서 장면 찾기 (매칭 모드)
def extract_scenes_by_matching(video_path, ppt_files, progress_bar):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps if fps > 0 else 0
    
    # PPT 이미지 미리 로드 및 전처리
    ppt_imgs = []
    ppt_filenames = []
    
    # 영상 크기에 맞춰 PPT 리사이징을 위해 첫 프레임 읽기
    ret, first_frame = cap.read()
    if not ret: return []
    h_vid, w_vid = first_frame.shape[:2]
    
    # 업로드된 PPT 읽어서 메모리에 올리기
    sorted_ppts = sorted(ppt_files, key=lambda x: x.name) # 이름순 정렬
    for p_file in sorted_ppts:
        # 파일 저장 후 읽기
        p_path = os.path.join(PPT_DIR, p_file.name)
        with open(p_path, "wb") as f: f.write(p_file.getbuffer())
        
        img = cv2.imread(p_path)
        if img is not None:
            # 영상 크기와 똑같이 리사이징 (비교를 위해)
            img_resized = cv2.resize(img, (w_vid, h_vid))
            gray_ppt = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            ppt_imgs.append(gray_ppt)
            ppt_filenames.append(p_file.name)

    if not ppt_imgs: return []

    scene_data = []
    current_ppt_idx = 0
    last_match_time = -999
    
    status = st.empty()
    status.write(f"🧩 PPT {len(ppt_imgs)}장과 영상 매칭 시작...")

    # 영상 스캔 (속도를 위해 0.5초 단위로 건너뛰며 스캔)
    step_frames = int(fps * 0.5) 
    
    while True:
        # 프레임 건너뛰기
        for _ in range(step_frames): cap.grab()
        ret, frame = cap.read()
        if not ret: break
        
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if duration > 0: progress_bar.progress(min(int((current_time/duration)*40), 40))

        # 현재 프레임 흑백 변환
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 현재 보고 있는 PPT와 다음 PPT랑 비교
        # 로직: "현재 PPT보다 다음 PPT랑 더 비슷해지면 넘어간 걸로 간주"
        
        score_current = 0
        score_next = 0
        
        # 현재 PPT와 유사도 (구조적 유사도 대신 간단히 픽셀 차이 역수 사용)
        diff_curr = np.mean(cv2.absdiff(frame_gray, ppt_imgs[current_ppt_idx]))
        score_current = 100 - diff_curr # 차이가 작을수록 점수 높음
        
        # 다음 PPT가 있다면 비교
        if current_ppt_idx < len(ppt_imgs) - 1:
            diff_next = np.mean(cv2.absdiff(frame_gray, ppt_imgs[current_ppt_idx+1]))
            score_next = 100 - diff_next
            
            # 다음 PPT랑 훨씬 더 비슷해지면 인덱스 변경 (장면 전환)
            # 10점 이상 차이나면 확실하게 넘어간 것
            if score_next > score_current + 10: 
                current_ppt_idx += 1
                
                # 결과 저장
                save_name = f"match_scene_{current_ppt_idx+1:02d}.jpg"
                save_path = os.path.join(OUTPUT_DIR, save_name)
                cv2.imwrite(save_path, frame) # 영상 프레임 저장
                
                # 혹은 원본 PPT를 결과로 쓰고 싶다면 아래 주석 해제
                # cv2.imwrite(save_path, cv2.imread(os.path.join(PPT_DIR, ppt_filenames[current_ppt_idx])))

                scene_data.append({
                    "seq": current_ppt_idx + 1,
                    "time": current_time,
                    "path": save_path,
                    "filename": save_name,
                    "ppt_source": ppt_filenames[current_ppt_idx]
                })
                status.write(f"✅ PPT {current_ppt_idx+1}번 매칭 성공! ({current_time:.1f}초)")

    cap.release()
    status.empty()
    
    # 첫 장면(PPT 1번)이 누락될 수 있으므로 강제 추가 (0초)
    if not scene_data:
        first_save = os.path.join(OUTPUT_DIR, "match_scene_01.jpg")
        cv2.imwrite(first_save, first_frame)
        scene_data.append({"seq": 1, "time": 0.0, "path": first_save, "filename": "match_scene_01.jpg"})
        
    return scene_data


# [Track 2] 자동 감지 모드 (기존 로직)
def extract_scenes_auto(video_path, sensitivity, cooldown, mask_dir, w_ratio, h_ratio, progress_bar):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps if fps > 0 else 0
    
    prev_frame = None
    last_capture_time = -cooldown
    scene_data = [] 
    scene_count = 0
    status_text = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret: break
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        if duration > 0 and int(current_time)%2==0:
            progress_bar.progress(min(int((current_time/duration)*40), 40))

        if current_time - last_capture_time < cooldown: continue

        h, w = frame.shape[:2]
        mask_w_px = int(w * (w_ratio / 100))
        mask_h_px = int(h * (h_ratio / 100))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        analyze_area = gray.copy()

        if mask_dir == "우측 하단": analyze_area[h-mask_h_px:h, w-mask_w_px:w] = 0
        elif mask_dir == "좌측 하단": analyze_area[h-mask_h_px:h, 0:mask_w_px] = 0
        elif mask_dir == "우측 상단": analyze_area[0:mask_h_px, w-mask_w_px:w] = 0

        is_changed = False
        if prev_frame is None: is_changed = True
        else:
            diff = np.mean(cv2.absdiff(prev_frame, analyze_area))
            if diff > sensitivity: is_changed = True

        if is_changed:
            scene_count += 1
            save_name = f"auto_scene_{scene_count:03d}.jpg"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            cv2.imwrite(save_path, frame)
            scene_data.append({"seq": scene_count, "time": current_time, "path": save_path, "filename": save_name})
            last_capture_time = current_time
            prev_frame = analyze_area
            status_text.write(f"📸 변화 감지: {scene_count}번 장면")

    cap.release()
    status_text.empty()
    return scene_data

# ==========================================
# 3. 공통 유틸리티 (초기화, Whisper, Gemini)
# ==========================================
def init_environment():
    try:
        for d in [TEMP_DIR, OUTPUT_DIR, PPT_DIR]:
            if os.path.exists(d): shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
    except: pass

def run_gemini(image_path, api_key):
    if not api_key: return "API 키 없음"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        img = genai.upload_file(image_path)
        return model.generate_content(["이 화면 요약", img]).text
    except Exception as e: return f"Gemini Error: {e}"

def run_whisper(video_path, api_key):
    if not api_key: return "API 키 없음"
    try:
        if os.path.exists(AUDIO_PATH): os.remove(AUDIO_PATH)
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(AUDIO_PATH, logger=None)
        clip.close()
        
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        with open(AUDIO_PATH, "rb") as f:
            return client.audio.transcriptions.create(model="whisper-1", file=f, response_format="text")
    except Exception as e: return f"Whisper Error: {e}"

def draw_mask_preview(frame, direction, w_ratio, h_ratio):
    preview = frame.copy()
    h, w = preview.shape[:2]
    mask_w, mask_h = int(w*(w_ratio/100)), int(h*(h_ratio/100))
    if direction == "우측 하단": cv2.rectangle(preview, (w-mask_w, h-mask_h), (w, h), (0,0,255), -1)
    elif direction == "좌측 하단": cv2.rectangle(preview, (0, h-mask_h), (mask_w, h), (0,0,255), -1)
    elif direction == "우측 상단": cv2.rectangle(preview, (w-mask_w, 0), (w, mask_h), (0,0,255), -1)
    return preview

# ==========================================
# 4. 메인 UI
# ==========================================
st.set_page_config(page_title="헌수학당 분석기 Final", layout="wide")
st.title("🎬 헌수학당 콘텐츠 분석기")

with st.sidebar:
    st.header("설정")
    openai_key = st.text_input("OpenAI Key", type="password")
    google_key = st.text_input("Gemini Key", type="password")
    st.divider()
    
    st.subheader("모드 설정")
    # 여기가 핵심입니다! PPT 유무에 따라 전략을 보여줍니다.
    mode_info = st.empty()
    
    st.divider()
    st.subheader("자동 감지 옵션 (PPT 없을 때만 사용)")
    sensitivity = st.slider("민감도", 5, 50, 15)
    cooldown = st.slider("최소 간격", 1.0, 5.0, 2.0)
    mask_dir = st.selectbox("가릴 위치", ["없음", "우측 하단", "좌측 하단", "우측 상단"])
    mask_w, mask_h = st.slider("가로 %", 0,50,20), st.slider("세로 %", 0,50,20)
    
    if st.button("🗑️ 초기화"):
        st.experimental_rerun()

col1, col2 = st.columns(2)
with col1:
    uploaded_video = st.file_uploader("1. 영상 파일", type=["mp4"])
    if uploaded_video:
        if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
        video_path = os.path.join(TEMP_DIR, "input.mp4")
        with open(video_path, "wb") as f: f.write(uploaded_video.getbuffer())
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            prev_img = draw_mask_preview(frame, mask_dir, mask_w, mask_h)
            st.image(cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB), caption="미리보기")

with col2:
    uploaded_ppts = st.file_uploader("2. PPT 이미지들 (매칭용)", accept_multiple_files=True)
    if uploaded_ppts:
        st.success(f"✅ PPT {len(uploaded_ppts)}장 로드됨! [Track 1: 매칭 모드]로 작동합니다.")
        mode_info.success("매칭 모드 활성화됨")
    else:
        st.info("PPT가 없습니다. [Track 2: 자동 감지 모드]로 작동합니다.")
        mode_info.info("자동 감지 모드")

st.divider()

if uploaded_video and st.button("🚀 분석 시작", type="primary"):
    init_environment()
    progress_bar = st.progress(0)
    video_path = os.path.join(TEMP_DIR, "input.mp4")
    # 파일 다시 확보 (초기화 대비)
    with open(video_path, "wb") as f: f.write(uploaded_video.getbuffer())
    
    # === [분기점] PPT가 있냐 없냐에 따라 다른 함수 호출 ===
    if uploaded_ppts:
        st.write("🔄 **Track 1 가동:** PPT 이미지를 기준으로 영상을 분석합니다...")
        scenes = extract_scenes_by_matching(video_path, uploaded_ppts, progress_bar)
    else:
        st.write("🎥 **Track 2 가동:** 화면 변화를 감지하여 영상을 분석합니다...")
        scenes = extract_scenes_auto(video_path, sensitivity, cooldown, mask_dir, mask_w, mask_h, progress_bar)
    
    if not scenes:
        st.error("장면 추출 실패. 설정을 확인하세요.")
        st.stop()
        
    st.success(f"Step 1 완료: {len(scenes)}개 장면")
    
    # Step 2: Gemini & Whisper (공통)
    st.info("AI 분석 시작...")
    final_data = []
    for i, s in enumerate(scenes):
        progress_bar.progress(40 + int((i/len(scenes))*50))
        desc = run_gemini(s['path'], google_key)
        final_data.append({"순서": s['seq'], "시간": f"{s['time']:.1f}", "설명": desc, "파일명": s['filename']})
    
    full_script = run_whisper(video_path, openai_key)
    progress_bar.progress(100)
    
    # 엑셀 저장
    df = pd.DataFrame(final_data)
    excel_path = "result.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='장면', index=False)
        pd.DataFrame({"스크립트": [full_script]}).to_excel(writer, sheet_name='스크립트', index=False)
        
    st.balloons()
    with open(excel_path, "rb") as f:
        st.download_button("📥 엑셀 다운로드", f, file_name="헌수학당_완성본.xlsx")
        
    # 결과 표시
    cols = st.columns(3)
    for i, row in df.iterrows():
        cols[i%3].image(os.path.join(OUTPUT_DIR, row['파일명']), caption=f"#{row['순서']}")