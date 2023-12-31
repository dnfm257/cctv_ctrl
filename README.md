# cctv_ctrl

## Team project

### Team: Watchdogs
### <<프로젝트 요약>>
#### 프로젝트 주제
* 다목적 CCTV 상황제어
거리에 있는 CCTV를 이용하여 트래픽에 따른 신호등 제어 및 각종 사건사고 감지

* 유스케이스
<img src=./doc/usecase.png>
* 흐름도

![diagram](https://github.com/dnfm257/cctv_ctrl/assets/118237074/2e4c1ce2-9671-4f5e-9acc-e2d6bbbaa88c)

* 시퀀스 다이어그램
```mermaid
sequenceDiagram

교통관제->>신호등: connect_client

loop traffic check
    교통관제->>교통관제: detect_traffic()
    교통관제->>신호등: send_msg()
end
```

```mermaid
sequenceDiagram
loop accident check
    교통관제->>교통관제: accident_detect()
    교통관제->>경찰: send_msg()
    교통관제->>병원: send_msg()
end
```

## 111702.py환경설정  
--- python3.8 사용 ---
```py
#yolov8s.pt설치방법
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```
```py
pip install opencv-python
pip install supervision
pip install YOLO
pip install numpy
```
## 112202.asm환경설정
--- pb9,pb13,pb15사용 ---
```c
sudo apt-get install gdb
sudo apt-get install gdb-multiarch
(GNU gdb (Ubuntu 12.1-0ubuntu1~22.04) 12.1)
sudo apt-get install openocd
(Open On-Chip Debugger 0.11.0, Licensed under GNU GPL v2)
sudo apt-get install stlink-tools
(stlink-server v2.1.1)
```
## stm32는 ide를 사용한 방법
aa.7z은 stm32 ide 전체파일 , import socket.py는 통신 기본 코드

## stm32 assembly를 통한 제어
### 112202.asm 사용법
```asm
#컴파일 방법
arm-none-eabi-as -mcpu=cortex-m4 -c file.asm -o file.o
arm-none-eabi-ld file.o -o file.elf -Ttext=0x08000000 
```
```asm
#작동방법
assembly coding
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg
#gdb port 변경
openocd -f interface/stlink.cfg -f target/stm32f1x.cfg -c "gdb_port port_number"
새로운 터미널 열고 gdb(gdb-multiarch)
load 파일위치/file.elf 또는 file 파일위치/file.elf
continue 또는 run
```
![스크린샷 2023-11-27 14-18-17](https://github.com/dnfm257/cctv_ctrl/assets/143377935/090bf48d-04f3-40ce-9076-e364a6f46a72)
```통합 명령
#컴파일 방법
arm-none-eabi -as -mcpu=cortex-m4 YourFileName.asm -o YourFileName.o && arm-none-eabi -ld YourFileName.o -o YourFileName.elf -Ttext=0x8000000 && openocd -f /YourPath/stlink.cfg -f /YourPath/stm32f4x.cfg -c "init; program start.elf; reset; exit;"
```

## final.py 실행
```python
pip install opencv-python
pip install "openvino>=2023.2.0"
pip install supervision
pip install ultralytics
```

```cmd
python ./final.py [CPU or GPU] [video_path] # default webcam 0
```



## 팀
* Members
  | Name | Role |
  |----|----|
  | 장석환 | edge_device_control, 신호등 STM32 제어 |
  | 김승현 | AI_modeling, 트래픽감지 ai 개발 및 학습 |
  | 김형은 | 문서 제작 및 ppt제작,발표 |
  | 서규승 | AI_modeling, 사고감지 ai 학습 및 project maneger |
  | 조성우 | edge_device_control, 신호등 및 raspberry cam제어 |
* Project Github : https://github.com/dnfm257/cctv_ctrl.git
* 발표자료 : https://github.com/dnfm257/cctv_ctrl/blob/main/doc/cctv_ctrl_ppt.pptx
