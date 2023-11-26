# STM32 NUCLEO-F429ZI를 assembly로 제어해본 내용
linux환경에서는 upgrade나 ubuntu version에 따라 일시적으로 stm32 ide프로그램이 작동하지 않을 수 있음
그러한 상황이 왔을때를 대비한 ide를 사용하지 않고 코딩하는 방법이 필요
다양한 방법 중 최소한의 파일이 필요한 assembly어를 선택하고 이를 시도해봄

## datasheet 
연산기호에 대한 내용(QRC0006_UAL16.pdf)
---
https://github.com/sysplay/bbb/blob/master/Docs/QRC0006_UAL16.pdf  
---

stm32 f429zi에 대한 datasheet(dm00031020-stm32f405-415-stm32f407-417-stm32f427-437-and-stm32f429-439-advanced-arm-based-32-bit-mcus-stmicroelectronics.pdf)
---
https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwit0IHP6-CCAxUsslYBHflPAgsQFnoECA0QAQ&url=https%3A%2F%2Fwww.st.com%2Fresource%2Fen%2Freference_manual%2Fdm00031020-stm32f405-415-stm32f407-417-stm32f427-437-and-stm32f429-439-advanced-arm-based-32-bit-mcus-stmicroelectronics.pdf&usg=AOvVaw2x8tbTRz8d9PfqXBk3qZ74&opi=89978449  
---
tim핀 설정을 위한 cortex-m4 datasheet
---
https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjq0piZ7OCCAxWsr1YBHYBYDbkQFnoECBAQAQ&url=https%3A%2F%2Fwww.st.com%2Fresource%2Fen%2Fprogramming_manual%2Fpm0214-stm32-cortexm4-mcus-and-mpus-programming-manual-stmicroelectronics.pdf&usg=AOvVaw0Wm2Kvb6UMOUO7rWx3Zup9&opi=89978449
---

MB1137.pdf  => f429zi의 pin에 대한 설정 파일(st.com)  

## 내장 LED제어(111d.asm)
--- 

