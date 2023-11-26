/*
 * app.c
 *
 *  Created on: Nov 23, 2023
 *      Author: iot22
 */

#include <stdio.h>
#include <stdbool.h>
#include "app.h"
#include "main.h"
#include "FreeRTOS.h"
#include "cmsis_os.h"
#include "lwip.h"
#include "lwip/api.h"
#include "lwip/ip_addr.h"
#include "lwip/arch.h"
#include "lwip/sockets.h"
int x,y=0,t=0;
char message[20];
int sock;
struct sockaddr_in server_addr;

extern UART_HandleTypeDef huart3;
void app_init(void);
void control(void)
{
	printf("gpio\r\n");
	HAL_GPIO_WritePin(GPIOC, GPIO_PIN_6, GPIO_PIN_SET);
	if(x>10)
	{
		osDelay(3000);
	}
	else if(10<x)
	{
		osDelay(4000);
	}
	HAL_GPIO_WritePin(GPIOC, GPIO_PIN_6, GPIO_PIN_RESET);
	printf("gpio_b\r\n");
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_15, GPIO_PIN_SET);
	osDelay(3000);
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_15, GPIO_PIN_RESET);
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_5, GPIO_PIN_SET);
	if(x<=10)
	{
		osDelay(3000);
	}
	else if(10<x)
	{
		osDelay(2000);
	}
	y=1;
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_5, GPIO_PIN_RESET);

}

void app(void)
{
		if(t==0)
		{
			app_init();
			t=1;
		}
		lwip_read(sock, message, 2);
		printf(message);
		printf("\r\n");
		x=atoi(message);
		if (x>5)
		{
			HAL_GPIO_WritePin(GPIOB, GPIO_PIN_12, GPIO_PIN_SET);
			osDelay(6000);
			HAL_GPIO_WritePin(GPIOB, GPIO_PIN_12, GPIO_PIN_RESET);
			x=15;
		}
		else
		{
			HAL_GPIO_WritePin(GPIOB, GPIO_PIN_12, GPIO_PIN_SET);
			osDelay(1000);
			HAL_GPIO_WritePin(GPIOB, GPIO_PIN_12, GPIO_PIN_RESET);
			x=1;
		}
		y=0;

}
void app_init(void)
{
	printf("app_init\r\n");
	sock = lwip_socket(AF_INET, SOCK_STREAM, 0);
	memset(&server_addr, 0, sizeof(server_addr));
	//HAL_GPIO_WritePin(GPIOC, GPIO_PIN_7, GPIO_PIN_SET);
	server_addr.sin_family = AF_INET;
	server_addr.sin_port = htons(5001);
	server_addr.sin_addr.s_addr = inet_addr("10.10.141.24");
	while(lwip_connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr))<0)
	{
		osDelay(3);
	}
	printf("app_inend\r\n");
}
__weak int __io_putchar(int ch)
{
	HAL_UART_Transmit(&huart3, (uint8_t *)&ch, 1, 0xffff);

	return ch;
}

