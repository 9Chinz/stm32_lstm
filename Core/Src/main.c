/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "crc.h"
#include "usb_device.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "usbd_cdc_if.h"

#include "lstm_model.h"
#include "lstm_model_data.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
char sendBuf[255];


/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
ai_handle network = AI_HANDLE_NULL;

ai_u8 activitions[AI_LSTM_MODEL_DATA_ACTIVATIONS_SIZE];
ai_u8 in_data[AI_LSTM_MODEL_IN_1_SIZE_BYTES];
ai_u8 out_data[AI_LSTM_MODEL_OUT_1_SIZE_BYTES];

ai_buffer *ai_input;
ai_buffer *ai_output;

#define NUM_INSTANCES 10

int sample_n = 0;

static ai_float* _lstm_states[NUM_INSTANCES] = {};
static ai_u32 _lstm_states_size[NUM_INSTANCES] = {};
static int _lstm_instance_idx = 0;
void _allocate_lstm_states(ai_float **states, ai_u32 size_in_bytes)
{
  if ((_lstm_instance_idx > NUM_INSTANCES) || (!states) || (*states) || (!size_in_bytes)) {
      /* error - invalid call */
	  sprintf(sendBuf, "Error - invalid call \r\n");
	  CDC_Transmit_FS((uint8_t *)sendBuf, strlen(sendBuf));
      return;
  }
  ai_handle src = AI_HANDLE_PTR(*states);
  _lstm_states[_lstm_instance_idx] = (ai_float *)malloc(size_in_bytes);
  _lstm_states_size[_lstm_instance_idx] = size_in_bytes;
  *states = _lstm_states[_lstm_instance_idx];
  /*
     Clear lstm initial state or
     set the state with a user-defined value.
  */
  if (*states) {
    memset(*states, 0, size_in_bytes);
  }
  _lstm_instance_idx++;
}

void clear_lstm_states(void) {
    for (int i=0; i<NUM_INSTANCES; i++) {
		memset(_lstm_states[i], 0, _lstm_states_size[i]);
    }
}

short select_in = 0;

/*
 * label has 2 class 0 is not fault 1 is fault
 * first input is 1
 * second is 0
 * when input must have x, y, z
 * and each of it must have 10 items then concat to make 1d array of x*10 + y*10 + z*10 items
 * total must equal to 30
 * input it's depends on sliding windows
 */
float test_input[2][30] = {
		{-0.33565952, -0.25572453,-1.45066868,-0.71036981,-1.45666981,0.00928518,
		1.16774249,-1.90891464,-1.04091182,-0.54233829, -0.74166618,-1.23648201,
		-1.79143868,-1.79673181,-1.73476575,-1.76168768, -1.56027518,-0.5909034,
		-0.34805852, 0.70198777,-0.66896681, 0.36941662,-0.37603663, 1.03044008,
		0.51871319, 0.19026073,-0.37346255, 0.55269103,-2.21135489 ,2.15170885
		},
		{0.03160937, 0.04025099,0.54986659,0.02656842,0.19796057,0.02920892,
		-0.19379291,0.78199013,0.5179406,-0.07016972,-0.05000088, -0.06067839,
		-0.02901091, -0.00281907, -0.03539916, 0.03039984, 0.01744082, -0.03366521,
		0.05394512, 0.02903093, -0.09340277, 0.13826433, 0.22012004, -0.26792532,
		-0.02081374, -0.14900287, -0.30087353, 0.23299044, -0.34720695, -0.28851795
		}
};
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_CRC_Init();
  MX_USB_DEVICE_Init();
  /* USER CODE BEGIN 2 */
  ai_error err;

  err = ai_lstm_model_create(&network, AI_LSTM_MODEL_DATA_CONFIG);

  if (err.type != AI_ERROR_NONE){
	  sprintf(sendBuf, "Error create \r\n");
	  CDC_Transmit_FS((uint8_t *)sendBuf, strlen(sendBuf));
  }
  ai_network_params ai_param = {
		  AI_LSTM_MODEL_DATA_WEIGHTS(ai_lstm_model_data_weights_get()),
		  AI_LSTM_MODEL_DATA_ACTIVATIONS(activitions)
  };

  if(!ai_lstm_model_init(network, &ai_param)){
	  sprintf(sendBuf, "Error init \r\n");
	  CDC_Transmit_FS((uint8_t *)sendBuf, strlen(sendBuf));
  }
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

	  sprintf(sendBuf, "Input %d \r\n", select_in);
	  CDC_Transmit_FS((uint8_t *)sendBuf, strlen(sendBuf));

	  /* 1 input */
	  for (uint8_t i = 0; i < AI_LSTM_MODEL_IN_1_SIZE; i++){
		 ((ai_float *)in_data)[i] = (ai_float)test_input[select_in][i];
	  }

	  ai_i32 n_batch;

	  ai_input = ai_lstm_model_inputs_get(network, NULL);
	  ai_output = ai_lstm_model_outputs_get(network, NULL);

	  ai_input[0].data = AI_HANDLE_PTR(in_data);
	  ai_output[0].data = AI_HANDLE_PTR(out_data);

	  /* 2 before run lstm */
	  if (sample_n++ > 20) {
		  clear_lstm_states();
		  sample_n = 0;
	  }

	  /* 3 run model */
	  n_batch = ai_lstm_model_run(network, &ai_input[0], &ai_output[0]);

	  // output is depends on num of class total of class now is 0, 1
	  for(uint8_t i = 0; i < AI_LSTM_MODEL_OUT_1_SIZE; i++){
//		  sprintf(sendBuf, "out label data %d data is %f\r\n", i, ((float *)out_data)[i]);
//		  CDC_Transmit_FS((uint8_t *) sendBuf, strlen(sendBuf));
		  if (((float *)out_data)[i] > 0.5){
			  sprintf(sendBuf, "  Output is class %d at prop %f \r\n", i, ((float *)out_data)[i]);
			  CDC_Transmit_FS((uint8_t *) sendBuf, strlen(sendBuf));
		  }
	  }

	  select_in++;
	  if(select_in > 1){
		  select_in = 0;
	  }

	  /* 3 run */
	  HAL_Delay(500);
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_MSI;
  RCC_OscInitStruct.MSIState = RCC_MSI_ON;
  RCC_OscInitStruct.MSICalibrationValue = 0;
  RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_6;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_MSI;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 32;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV7;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
