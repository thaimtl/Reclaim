const uint8_t PINS[] = {A0, A1, A2, A3};
const uint8_t N = sizeof PINS;
const uint32_t PERIOD_US = 1000;

void setup() {
  Serial.begin(250000);
}

void loop() {
  static uint32_t next = micros();
  static uint8_t count = 0;
  while ((int32_t)(micros() - next) < 0) {}
  next += PERIOD_US;

  uint8_t frame[1 + 2 * N];
  frame[0] = 0x80 | (count++ & 0x7F);
  for (uint8_t i = 0; i < N; i++) {
    uint16_t v = analogRead(PINS[i]);
    frame[1 + 2 * i] = v >> 5;
    frame[2 + 2 * i] = v & 0x1F;
  }
  Serial.write(frame, sizeof frame);
}
