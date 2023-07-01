#include <Encoder.h>
#include <Wire.h>
#include "Adafruit_TCS34725.h"
#include <MPU6050_light.h>
#include "Adafruit_VL53L0X.h"
#include <Servo.h> 
 
Servo lift_servo; 
Servo gr_L;
Servo gr_R;
Servo drop_R;
Servo drop_L;

MPU6050 mpu(Wire);

#define left_ir A2
#define right_ir A1
#define cytronL A3
#define cytronR A0

#define motor_A_FR 1
#define motor_B_FR 0

#define motor_A_FL 2
#define motor_B_FL 3

#define motor_A_BL 6
#define motor_B_BL 4

#define motor_A_BR 5
#define motor_B_BR 7

Encoder encFR(26, 27);
Encoder encFL(29, 28);
Encoder encBL(33, 30);
Encoder encBR(32, 31);
Adafruit_TCS34725 colourL = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_2_4MS, TCS34725_GAIN_16X);
Adafruit_TCS34725 colourR = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_2_4MS, TCS34725_GAIN_16X);

Adafruit_VL53L0X lefttof = Adafruit_VL53L0X();
Adafruit_VL53L0X righttof = Adafruit_VL53L0X();
Adafruit_VL53L0X fronttof = Adafruit_VL53L0X();

int target = 1, steering_inp = 0, right_count = 0, left_count = 0;
float out = 1;
float calibrate_val = 0, sum_calibrate = 0, deltaT = 12.2;
bool flag_main = true;

long prev_timeFR = 0, timeFR = 1, oldPosFR = -999, newPosFR =1;
float kpFR=10;

long prev_timeFL = 0, timeFL = 1, oldPosFL = -999, newPosFL =1;
float kpFL = 10;

long prev_timeBL = 0, timeBL = 1, oldPosBL = -999, newPosBL = 1;
float kpBL = 10;

long prev_timeBR = 0, timeBR = 1, oldPosBR = -999, newPosBR = 1;
float kpBR = 10;

int targetL = 0;
int targetR = 0;
float left_ir_val = 0, right_ir_val = 0;

int redR_measure = 32, greenR_measure = 55, blueR_measure = 40;
int redL_measure = 37, greenL_measure = 60, blueL_measure = 43;

int white_left = 961, white_right = 965, black_left = 1014, black_right = 1013;

int flagDir = 0, abscissa = 0;
String readpi;

int countOb1 = 0;
int fronttof_val = 1000;
float e=0;
int lefttof_val = 0;
int righttof_val = 0;

void setup() {
  Wire.begin();
  Serial.begin(9600);
  Serial5.begin(9600);
  delay(500);
  Serial5.flush();
  delay(500);
  Serial5.print("reset");
  delay(500);
  pinMode(motor_A_FR, OUTPUT);
  pinMode(motor_B_FR, OUTPUT);
  pinMode(motor_A_FL, OUTPUT);
  pinMode(motor_B_FL, OUTPUT);
  pinMode(motor_A_BR, OUTPUT);
  pinMode(motor_B_BR, OUTPUT);
  pinMode(motor_A_BL, OUTPUT);
  pinMode(motor_B_BL, OUTPUT);
  pinMode(13, OUTPUT);

  pinMode(23, INPUT);
  pinMode(22, INPUT);

  
  lift_servo.attach(8); 
  gr_L.attach(11);
  gr_R.attach(12);
  drop_R.attach(10);
  drop_L.attach(9);
  
  lift_servo.write(30);      // 30-150
  gr_L.write(30);            
  gr_R.write(140);
  
  drop_R.write(80);        //open: 180, 70    closed: 80, 170
  drop_L.write(170);
  
  
  calibration();

}

void loop() {
 evac();
 obst_L_1();
 green_red();
 Basic_LF(1.5,0, 40);

}
