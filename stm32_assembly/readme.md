# STM32 NUCLEO-F429ZI를 assembly로 제어해본 내용
linux환경에서는 upgrade나 ubuntu version에 따라 일시적으로 stm32 ide프로그램이 작동하지 않을 수 있음
그러한 상황이 왔을때를 대비한 ide를 사용하지 않고 코딩하는 방법이 필요
다양한 방법 중 최소한의 파일이 필요한 assembly어를 선택하고 이를 시도해봄

## 데이터에 관련된 내용
datasheet폴더 
연산기호에 대한 내용(QRC0006_UAL16.pdf) = https://github.com/sysplay/bbb/blob/master/Docs/QRC0006_UAL16.pdf  

MB1137.pdf  => f429zi의 pin에 대한 설정 파일(st.com)  

## 내장 LED제어(111d.asm)
--- 

