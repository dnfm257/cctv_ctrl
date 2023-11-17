# cctv_ctrl

## Team project

### Team: Watchdogs
<프로젝트 요약>
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

* Members
  | Name | Role |
  |----|----|
  | 장석환 | Project lead, 프로젝트 총괄 및 책임 |
  | 김승현 | Project manager, github repository 생성 및 프로젝트 이슈 진행상황 관리 |
  | 김형은 | UI design, 사용자 인터페이스 정의 및 구현 |
  | 서규승 | AI modeling, AI model 선택, data 수집 및 training 수행 |
  | 조성우 | Architect, 프로젝트 component 구성 및 상위 디자인 설계 |
* Project Github : https://github.com/dnfm257/cctv_ctrl.git
* 발표자료 : -
