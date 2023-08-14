#include <AFMotor.h>
#include <Servo.h>
#include <ArduinoJson.h>
#include <LiquidCrystal_I2C.h>

AF_Stepper motor_step(200,1);
AF_Stepper motor_act(60,2);
int flag = 0;
Servo myservo;
LiquidCrystal_I2C lcd(0x27, 16, 2);
StaticJsonDocument<42> doc;

// 모터 설정
void setup() {
  lcd.init(); 
  lcd.backlight();
  motor_step.setSpeed(8); // 스텝 모터 속도
  motor_step.onestep(FORWARD, DOUBLE); // 스텝 모터 동작
  motor_step.release(); // 스텝 모터 정지

  motor_act.setSpeed(480);  // 액추에이더 속도 설정
  motor_act.onestep(FORWARD, INTERLEAVE); // 액추에이터 전진
  motor_act.release(); // 액추에이터 정지

  Serial.begin(115200); // 시리얼 통신 속도, 젯슨나노와 동일
  myservo.attach(10);

  delay(1000);
  
}

void doStep(){
  doc["STEP"] = 1; // JSON 문서 doc에 STEP 키를 추가하고 값을 1로 설정
  serializeJson(doc, Serial); // JSON 문서를 시리얼 포트로 전송(젯슨 나노로 전송)
  Serial.println(); // 개행을 위한 코드
  motor_step.step(40,FORWARD,DOUBLE); // 스텝 모터 40단계만큼 전진, 회전부가 5각이므로
}

void doAction(){
  while(Serial.available() == 0){

  } // 젯슨나노로부터 시리얼포트로 전송된 데이터를 받을 때까지 대기, blocking 코드
  
  deserializeJson(doc, Serial); // 읽어온 JSON 데이터를 읽어와서 역직렬화 
  int act_flag = doc["act"]; // act 키에 해당하는 값을 act_flag에 저장, NORMAL = 1, BROKEN = 2
  float accuracy = doc["accuracy"]; // accuracy 키에 해당하는 값을 accuracy에 저장, 모델의 확신율

  // 젯슨나노로부터 받은 모델 클래스 결과값의 플래그 값을 이용하여 센서 동작
  if(act_flag == 1){ // 골프공이 NORMAL 인 경우
    lcd.setCursor(0,0);
    lcd.print("Pred : NORMAL"); // LCD에 출력
    lcd.setCursor(0,1);
    lcd.print(accuracy,4); // 확신율을 소수점 넷째짜리까지 표현
  } else if(act_flag == 2){ // 골프공이 BROKEN인 경우
    lcd.setCursor(0,0);
    lcd.print("Pred : BROKEN");
    lcd.setCursor(0,1);
    lcd.print(accuracy,4);
  }
  motor_act.step(900, FORWARD, INTERLEAVE); // 타격부에 떨어진 골프공을 액추에이터를 움직여 분류부로 보내는 동작
  delay(50);
  motor_act.step(900, BACKWARD, INTERLEAVE);

  
  // 분류부 서보모터 동작 코드
  if(act_flag == 1){ // normal ball
    myservo.write(80);
    delay(2000);
    myservo.write(90);
    delay(1000);
    lcd.clear();
  } else if(act_flag == 2){ // broken ball
    myservo.write(100);
    delay(2000);
    myservo.write(90);
    delay(1000);
    lcd.clear();
  }

  doc.clear();
}
void loop() {
  doc.clear();
  doStep();
  doAction();
}
  
