int val =0;
bool band = false;
// Definimos los pines donde tenemos conectadas las bobinas
#define IN1  8
#define IN2  9
#define IN3  10
#define IN4  11


// Secuencia de pasos (par máximo)
int paso [4][4] =
{
  
  {1, 1, 0, 0},
  {0, 1, 1, 0},
  {0, 0, 1, 1},
  {1, 0, 0, 0},


};

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  Serial.begin(9600);// puerto en 9600 baudios
}

void loop() {
   if (band == true){
   
   for (int j = 0; j < 512; j++){  
                                   
   for (int i = 0; i < 4; i++)
    {
      digitalWrite(IN1, paso[i][0]); //4 pasos
      digitalWrite(IN2, paso[i][1]);
      digitalWrite(IN3, paso[i][2]);
      digitalWrite(IN4, paso[i][3]);
      //Serial.write(".");
      delay(10);
    }  
   }
    band = false;
    Serial.write("e");  
   }
}


void serialEvent(){
  if(Serial.available()){
    val = Serial.parseInt();
    if(val==1){
      band = true;
    }if(val==2){
      band = false;
    }
  }
}
