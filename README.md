# cctv_ctrl

## Team project

### Team: Watchdogs
### <<프로젝트 요약>>
#### 프로젝트 주제
* 다목적 CCTV 상황제어
거리에 있는 CCTV를 이용하여 트래픽에 따른 신호등 제어 및 각종 사건사고 감지

* 유스케이스
<img src=./doc/usecase.PNG>

* 클래스 다이어그램
<img src=./doc/class_diagram.PNG>

* 시퀀스 다이어그램
```mermaid
sequenceDiagram

CCTV->>교통관제: send_img()
Note right of 교통관제: 1 Cycle
교통관제->>교통관제: traffic_detect()

교통관제->>신호등: light_control()
신호등-->>교통관제: recv_cycle()
```

```mermaid
sequenceDiagram
loop accident check
    CCTV->>교통관제: send_img()
    교통관제->>교통관제: accident_detect()
end
교통관제->>경찰: emergency_signal()
경찰-->>교통관제: recv_msg()
교통관제->>병원: emergency_signal()
병원-->>교통관제: recv_msg()
```

```mermaid
sequenceDiagram
loop violation check
    CCTV->>교통관제: send_img()
    교통관제->>교통관제: violation_detect()
end
교통관제->>경찰: call_signal()
경찰-->>교통관제: recv_msg()
```
## 111702.py환경설정  
---python3.8 사용---
```py
pip install opencv-python
pip install supervision
pip install YOLO
pip install numpy
```
## 팀
* Members
  | Name | Role |
  |----|----|
  | 장석환 | AI_modeling, 사고분석 ai 학습 및 개발1 |
  | 김승현 | AI_modeling, 사고분석 ai 학습 및 개발2 |
  | 김형은 | 문서 제작 및 ppt제작,발표|
  | 서규승 | AI_modeling, 교통통제 및 project maneger|
  | 조성우 | edge_device_control, 신호등 및 raspberry cam제어 |
* Project Github : https://github.com/dnfm257/cctv_ctrl.git
* 발표자료 : https://github.com/dnfm257/cctv_ctrl/blob/main/doc/cctv_ctrl_ppt.pptx
