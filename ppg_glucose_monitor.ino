/*
 * ═══════════════════════════════════════════════════════════════════
 *  PPG Glucose Monitor — ESP32 Firmware
 *  Hardware: ESP32 + MAX30102 + SSD1306 OLED (128x64 I2C)
 *
 *  Wiring:
 *    MAX30102  SDA  → GPIO21
 *    MAX30102  SCL  → GPIO22
 *    MAX30102  VIN  → 3.3V
 *    MAX30102  GND  → GND
 *
 *    OLED SDA       → GPIO21  (shared I2C bus)
 *    OLED SCL       → GPIO22  (shared I2C bus)
 *    OLED VCC       → 3.3V
 *    OLED GND       → GND
 *
 *    Button         → GPIO4 (one side) + GND (other side)
 *
 *  Libraries needed (Arduino Library Manager):
 *    - Adafruit SSD1306
 *    - Adafruit GFX Library
 *    - ArduinoJson by Benoit Blanchon
 * ═══════════════════════════════════════════════════════════════════
 */

#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <ArduinoJson.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_GFX.h>

// ── WiFi credentials ──────────────────────────────────────────────
const char* WIFI_SSID     = "chiki chiki";       // ← change
const char* WIFI_PASSWORD = "thuwa567891011";   // ← change

// ── API endpoints ─────────────────────────────────────────────────
const char* API_URL    = "https://non-invasive-glucose-backend.onrender.com/predict";
const char* HEALTH_URL = "https://non-invasive-glucose-backend.onrender.com/health";

// Used by the raw streaming POST in sendPredictRequest() — avoids
// buffering the whole ~150KB JSON body in one contiguous allocation,
// which heap fragmentation causes to fail even with plenty of nominal
// free heap on plain (non-PSRAM) ESP32 boards.
const char*    API_HOST = "non-invasive-glucose-backend.onrender.com";
const char*    API_PATH = "/predict";
const uint16_t API_PORT = 443;

// ── Patient config ────────────────────────────────────────────────
const float PATIENT_AGE = 25.0;   // ← update per patient

// ── Pin definitions ───────────────────────────────────────────────
#define BUTTON_PIN    4

// ── OLED config ───────────────────────────────────────────────────
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
#define OLED_ADDR     0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ── MAX30102 registers ────────────────────────────────────────────
#define MAX30102_ADDR     0x57
#define REG_INTR_STATUS1  0x00
#define REG_INTR_ENABLE1  0x02
#define REG_FIFO_WR_PTR   0x04
#define REG_OVF_COUNTER   0x05
#define REG_FIFO_RD_PTR   0x06
#define REG_FIFO_DATA     0x07
#define REG_FIFO_CONFIG   0x08
#define REG_MODE_CONFIG   0x09
#define REG_SPO2_CONFIG   0x0A
#define REG_LED1_PA       0x0C
#define REG_LED2_PA       0x0D
#define FIFO_CONFIG_VAL   0x10
#define MODE_SPO2         0x03
#define SPO2_CONFIG_VAL   0x25
#define LED1_CURRENT      0x3F
#define LED2_CURRENT      0x1F
#define FINGER_THRESHOLD  50000UL

// ── Signal config ─────────────────────────────────────────────────
#define SAMPLE_RATE       100
#define COLLECT_SECONDS   60
#define TOTAL_SAMPLES     (SAMPLE_RATE * COLLECT_SECONDS)  // 6000

// ── DC removal ────────────────────────────────────────────────────
float irDC  = 0;
float redDC = 0;
const float ALPHA_IR  = 0.97;
const float ALPHA_RED = 0.97;

// ── Smoothing buffer ──────────────────────────────────────────────
#define SMOOTH_SIZE 12
float irSmooth[SMOOTH_SIZE];
float redSmooth[SMOOTH_SIZE];
int   smoothIndex = 0;

// ── Signal buffers ────────────────────────────────────────────────
float* irBuffer  = nullptr;
float* redBuffer = nullptr;
int    sampleCount = 0;

// ── State machine ─────────────────────────────────────────────────
enum State {
  STATE_IDLE,
  STATE_ACQUIRE,
  STATE_SEND,
  STATE_RESULT,
  STATE_ERROR
};
State currentState = STATE_IDLE;

// ═══════════════════════════════════════════════════════════════════
//  MAX30102 FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MAX30102_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

uint8_t readReg(uint8_t reg) {
  Wire.beginTransmission(MAX30102_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MAX30102_ADDR, (uint8_t)1);
  return Wire.read();
}

bool readFIFOSample(long &red, long &ir) {
  Wire.beginTransmission(MAX30102_ADDR);
  Wire.write(REG_FIFO_DATA);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MAX30102_ADDR, (uint8_t)6);
  if (Wire.available() < 6) return false;

  uint32_t r = Wire.read() << 16;
  r |= Wire.read() << 8;
  r |= Wire.read();
  red = r & 0x3FFFF;

  uint32_t i = Wire.read() << 16;
  i |= Wire.read() << 8;
  i |= Wire.read();
  ir = i & 0x3FFFF;
  return true;
}

int fifoAvailable() {
  uint8_t wr = readReg(REG_FIFO_WR_PTR) & 0x1F;
  uint8_t rd = readReg(REG_FIFO_RD_PTR) & 0x1F;
  return (wr - rd + 32) & 0x1F;
}

void resetSensor() {
  writeReg(REG_MODE_CONFIG, 0x40);
  delay(50);
}

void initSensor() {
  resetSensor();
  writeReg(REG_FIFO_WR_PTR,  0x00);
  writeReg(REG_OVF_COUNTER,  0x00);
  writeReg(REG_FIFO_RD_PTR,  0x00);
  writeReg(REG_FIFO_CONFIG,  FIFO_CONFIG_VAL);
  writeReg(REG_MODE_CONFIG,  MODE_SPO2);
  writeReg(REG_SPO2_CONFIG,  SPO2_CONFIG_VAL);
  writeReg(REG_LED1_PA,      LED1_CURRENT);
  writeReg(REG_LED2_PA,      LED2_CURRENT);
  writeReg(REG_INTR_ENABLE1, 0x40);
}

void clearFIFO() {
  writeReg(REG_FIFO_WR_PTR, 0x00);
  writeReg(REG_OVF_COUNTER, 0x00);
  writeReg(REG_FIFO_RD_PTR, 0x00);
}

float smoothSignal(float* arr, float newVal) {
  arr[smoothIndex % SMOOTH_SIZE] = newVal;
  float sum = 0;
  for (int i = 0; i < SMOOTH_SIZE; i++) sum += arr[i];
  return sum / SMOOTH_SIZE;
}

void resetSignalState() {
  irDC = 0; redDC = 0;
  smoothIndex = 0;
  for (int i = 0; i < SMOOTH_SIZE; i++) {
    irSmooth[i] = 0;
    redSmooth[i] = 0;
  }
  clearFIFO();
}

// ═══════════════════════════════════════════════════════════════════
//  OLED DISPLAY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

void displayClear() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
}

void showIdle() {
  displayClear();
  display.setTextSize(1);
  display.setCursor(15, 0);
  display.println("GLUCOSE MONITOR");
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);
  display.setCursor(10, 20);
  display.println("Place finger &");
  display.setCursor(10, 32);
  display.println("press button");
  display.setCursor(10, 44);
  display.println("to start...");
  display.display();
}

void showFingerWait() {
  displayClear();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("No finger detected");
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);
  display.setCursor(5, 22);
  display.println("Place finger firmly");
  display.setCursor(5, 36);
  display.println("on the sensor...");
  display.display();
}

void showAcquiring(int remaining, int collected) {
  displayClear();
  display.setTextSize(1);
  display.setCursor(20, 0);
  display.println("Acquiring PPG");
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  // Big countdown timer
  display.setTextSize(3);
  if (remaining < 10) display.setCursor(52, 18);
  else display.setCursor(40, 18);
  display.print(remaining);
  display.setTextSize(1);
  display.setCursor(90, 28);
  display.print("sec");

  // Progress bar
  display.drawRect(0, 50, 128, 10, SSD1306_WHITE);
  int progress = map(collected, 0, TOTAL_SAMPLES, 0, 126);
  display.fillRect(1, 51, progress, 8, SSD1306_WHITE);

  display.display();
}

void showSending() {
  displayClear();
  display.setTextSize(1);
  display.setCursor(20, 0);
  display.println("Processing...");
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);
  display.setCursor(5, 22);
  display.println("Sending signal");
  display.setCursor(5, 34);
  display.println("to cloud server...");
  display.setCursor(5, 48);
  display.println("Please wait");
  display.display();
}

void showWakingServer() {
  displayClear();
  display.setTextSize(1);
  display.setCursor(15, 0);
  display.println("Waking server...");
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);
  display.setCursor(5, 22);
  display.println("Server may take");
  display.setCursor(5, 34);
  display.println("30-60 seconds");
  display.setCursor(5, 48);
  display.println("Please wait...");
  display.display();
}

void showResult(float glucose, String zone) {
  displayClear();
  display.setTextSize(1);
  display.setCursor(20, 0);
  display.println("RESULT");
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  // Glucose value — large text
  display.setTextSize(2);
  display.setCursor(5, 16);
  display.print(glucose, 1);
  display.setTextSize(1);
  display.setCursor(90, 22);
  display.println("mg/dL");

  display.drawLine(0, 38, 127, 38, SSD1306_WHITE);

  // Zone
  display.setTextSize(1);
  display.setCursor(5, 44);
  display.print("Status: ");
  display.println(zone);

  display.display();
}

void showError(String msg) {
  displayClear();
  display.setTextSize(1);
  display.setCursor(30, 0);
  display.println("ERROR");
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);
  display.setCursor(0, 20);
  display.println(msg);
  display.setCursor(0, 50);
  display.println("Press button retry");
  display.display();
}

void showWiFiConnecting() {
  displayClear();
  display.setTextSize(1);
  display.setCursor(5, 0);
  display.println("Connecting WiFi...");
  display.setCursor(5, 16);
  display.println(WIFI_SSID);
  display.display();
}

void showWiFiConnected() {
  displayClear();
  display.setTextSize(1);
  display.setCursor(15, 0);
  display.println("WiFi Connected!");
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);
  display.setCursor(5, 20);
  display.println(WiFi.localIP().toString());
  display.display();
  delay(1500);
}

// ═══════════════════════════════════════════════════════════════════
//  WIFI FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

bool connectWiFi() {
  showWiFiConnecting();
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    showWiFiConnected();
    return true;
  }
  showError("WiFi Failed!\nCheck credentials");
  return false;
}

bool wakeUpServer() {
  showWakingServer();
  Serial.println("Pinging server...");
  HTTPClient http;
  http.begin(HEALTH_URL);
  http.setTimeout(60000);
  int code = http.GET();
  http.end();
  if (code == 200) {
    Serial.println("Server awake!");
    return true;
  }
  Serial.println("Server ping failed: " + String(code));
  return false;
}

// ═══════════════════════════════════════════════════════════════════
//  STREAMING PREDICT REQUEST
//
//  The JSON body for /predict is ~150KB (2 * TOTAL_SAMPLES floats).
//  Building it as one String/buffer needs one large contiguous heap
//  block, which fails on plain (non-PSRAM) ESP32 boards once WiFi/TLS
//  have fragmented the heap — even with plenty of nominal free heap.
//  Instead, this computes the exact Content-Length with a cheap
//  "dry run" (snprintf with a throwaway buffer just to get the length),
//  then streams the body straight to the TLS socket a few bytes at a
//  time, so peak extra RAM use is a handful of bytes, not 150KB.
// ═══════════════════════════════════════════════════════════════════

size_t irRedJsonLength() {
  char tmp[24];
  size_t total = 0;
  total += snprintf(tmp, sizeof(tmp), "{\"age\":%.1f,\"ir\":[", PATIENT_AGE);
  for (int i = 0; i < TOTAL_SAMPLES; i++) {
    total += snprintf(tmp, sizeof(tmp), i < TOTAL_SAMPLES - 1 ? "%.4f," : "%.4f", irBuffer[i]);
  }
  total += snprintf(tmp, sizeof(tmp), "],\"red\":[");
  for (int i = 0; i < TOTAL_SAMPLES; i++) {
    total += snprintf(tmp, sizeof(tmp), i < TOTAL_SAMPLES - 1 ? "%.4f," : "%.4f", redBuffer[i]);
  }
  total += snprintf(tmp, sizeof(tmp), "]}");
  return total;
}

bool sendPredictRequest(int &httpCode, String &responseBody) {
  WiFiClientSecure client;
  client.setInsecure();          // no cert pinning, same as HTTPClient's default https behavior
  client.setTimeout(30);

  Serial.printf("Free heap before send: %u bytes\n", ESP.getFreeHeap());
  Serial.println("Connecting to API host...");
  if (!client.connect(API_HOST, API_PORT)) {
    Serial.println("Connection to API host failed!");
    return false;
  }

  size_t contentLength = irRedJsonLength();
  Serial.printf("Computed Content-Length: %u bytes\n", (unsigned)contentLength);

  client.printf("POST %s HTTP/1.1\r\n", API_PATH);
  client.printf("Host: %s\r\n", API_HOST);
  client.println("Content-Type: application/json");
  client.printf("Content-Length: %u\r\n", (unsigned)contentLength);
  client.println("Connection: close");
  client.println();

  // ── Stream body — never holds more than a few bytes at a time ──
  char buf[24];
  int  n;

  n = snprintf(buf, sizeof(buf), "{\"age\":%.1f,\"ir\":[", PATIENT_AGE);
  client.write((const uint8_t*)buf, n);

  for (int i = 0; i < TOTAL_SAMPLES; i++) {
    n = snprintf(buf, sizeof(buf), i < TOTAL_SAMPLES - 1 ? "%.4f," : "%.4f", irBuffer[i]);
    client.write((const uint8_t*)buf, n);
    if (i % 500 == 0) {
      yield();
      Serial.printf("Streaming IR: %d/%d\n", i, TOTAL_SAMPLES);
    }
  }

  n = snprintf(buf, sizeof(buf), "],\"red\":[");
  client.write((const uint8_t*)buf, n);

  for (int i = 0; i < TOTAL_SAMPLES; i++) {
    n = snprintf(buf, sizeof(buf), i < TOTAL_SAMPLES - 1 ? "%.4f," : "%.4f", redBuffer[i]);
    client.write((const uint8_t*)buf, n);
    if (i % 500 == 0) yield();
  }

  n = snprintf(buf, sizeof(buf), "]}");
  client.write((const uint8_t*)buf, n);

  Serial.println("Body sent, waiting for response...");

  // ── Wait for response ────────────────────────────────────────────
  unsigned long deadline = millis() + 60000UL;
  while (client.connected() && !client.available()) {
    if (millis() > deadline) {
      Serial.println("Timed out waiting for response");
      client.stop();
      return false;
    }
    delay(10);
  }

  String statusLine = client.readStringUntil('\n');
  Serial.println("Status line: " + statusLine);
  int sp1  = statusLine.indexOf(' ');
  httpCode = (sp1 >= 0) ? statusLine.substring(sp1 + 1, sp1 + 4).toInt() : 0;

  // Read response headers, capturing Content-Length / chunked so we
  // know exactly how many body bytes to expect — relying on the
  // socket closing (as the old code did) can hang for a long time if
  // Render's proxy keeps the connection open, and silently corrupts
  // the JSON if the response is chunked instead of a flat body.
  long contentLenResp = -1;
  bool chunked = false;
  while (client.connected() || client.available()) {
    String line = client.readStringUntil('\n');
    if (line.length() <= 1) break;   // bare "\r" == end of headers
    line.trim();
    String lower = line;
    lower.toLowerCase();
    if (lower.startsWith("content-length:")) {
      contentLenResp = line.substring(line.indexOf(':') + 1).toInt();
    } else if (lower.startsWith("transfer-encoding:") && lower.indexOf("chunked") >= 0) {
      chunked = true;
    }
  }

  responseBody = "";
  unsigned long readDeadline = millis() + 20000UL;   // 20s to read the (small) body

  if (chunked) {
    while (millis() < readDeadline) {
      while (!client.available() && client.connected() && millis() < readDeadline) delay(5);
      String sizeLine = client.readStringUntil('\n');
      sizeLine.trim();
      long chunkSize = strtol(sizeLine.c_str(), nullptr, 16);
      if (chunkSize <= 0) break;   // terminating 0-size chunk
      long got = 0;
      while (got < chunkSize && millis() < readDeadline) {
        while (!client.available() && client.connected() && millis() < readDeadline) delay(5);
        while (client.available() && got < chunkSize) {
          responseBody += (char)client.read();
          got++;
        }
      }
      client.readStringUntil('\n');   // consume the CRLF after chunk data
    }
  } else if (contentLenResp >= 0) {
    long got = 0;
    while (got < contentLenResp && millis() < readDeadline) {
      while (!client.available() && client.connected() && millis() < readDeadline) delay(5);
      while (client.available() && got < contentLenResp) {
        responseBody += (char)client.read();
        got++;
      }
    }
  } else {
    // No length info at all — fall back to reading until close,
    // but still bounded by the deadline so we can't hang forever.
    while ((client.connected() || client.available()) && millis() < readDeadline) {
      while (client.available()) responseBody += (char)client.read();
    }
  }

  client.stop();
  Serial.printf("Response body read: %u bytes\n", responseBody.length());

  return true;
}

// ═══════════════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);
  Serial.println("PPG Glucose Monitor starting...");

  // Button
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  // I2C
  Wire.begin(21, 22);
  Wire.setClock(400000);

  // OLED init
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("OLED init failed!");
    while (true);
  }
  display.clearDisplay();
  display.display();

  // Splash screen
  displayClear();
  display.setTextSize(1);
  display.setCursor(10, 10);
  display.println("PPG Glucose");
  display.setCursor(25, 25);
  display.println("Monitor v1.0");
  display.setCursor(15, 45);
  display.println("Initializing...");
  display.display();
  delay(1500);

  // MAX30102 init
  initSensor();
  Serial.println("MAX30102 initialized");

  // Allocate signal buffers
  irBuffer  = new float[TOTAL_SAMPLES];
  redBuffer = new float[TOTAL_SAMPLES];
  if (!irBuffer || !redBuffer) {
    Serial.println("Memory allocation failed!");
    showError("Memory error!\nRestart device");
    while (true);
  }

  // Init smoothing arrays
  for (int i = 0; i < SMOOTH_SIZE; i++) {
    irSmooth[i] = 0;
    redSmooth[i] = 0;
  }

  // Connect WiFi
  if (!connectWiFi()) {
    while (true) delay(1000);
  }

  // Wake up Render server
  wakeUpServer();

  // Show idle screen
  showIdle();
  Serial.println("Ready — press button to start");
}

// ═══════════════════════════════════════════════════════════════════
//  LOOP
// ═══════════════════════════════════════════════════════════════════

void loop() {

  // ── STATE: IDLE ────────────────────────────────────────────────
  if (currentState == STATE_IDLE) {
    if (digitalRead(BUTTON_PIN) == LOW) {
      delay(50);   // debounce
      if (digitalRead(BUTTON_PIN) == LOW) {
        Serial.println("Button pressed — starting acquisition");

        // Wait for button release
        while (digitalRead(BUTTON_PIN) == LOW) delay(10);

        // Reset everything
        sampleCount = 0;
        resetSignalState();

        // Check finger presence first
        showFingerWait();
        unsigned long fingerWaitStart = millis();
        bool fingerDetected = false;

        while (millis() - fingerWaitStart < 10000) {
          readReg(REG_INTR_STATUS1);
          int avail = fifoAvailable();
          for (int s = 0; s < avail; s++) {
            long irRaw = 0, redRaw = 0;
            if (readFIFOSample(redRaw, irRaw)) {
              if (irRaw >= FINGER_THRESHOLD) {
                fingerDetected = true;
                break;
              }
            }
          }
          if (fingerDetected) break;
          delay(100);
        }

        if (!fingerDetected) {
          showError("No finger!\nPlace & retry");
          delay(3000);
          showIdle();
          return;
        }

        Serial.println("Finger detected — acquiring...");
        currentState = STATE_ACQUIRE;
      }
    }
    return;
  }

  // ── STATE: ACQUIRE ─────────────────────────────────────────────
  if (currentState == STATE_ACQUIRE) {
    unsigned long startTime   = millis();
    unsigned long lastDisplay = 0;
    unsigned long lastSample  = millis();
    long          sampleInterval = 1000 / SAMPLE_RATE;  // 10ms

    Serial.printf("Collecting %d samples at %dHz...\n",
                  TOTAL_SAMPLES, SAMPLE_RATE);

    while (sampleCount < TOTAL_SAMPLES) {
      unsigned long now = millis();

      // Update display every second
      if (now - lastDisplay >= 1000) {
        lastDisplay   = now;
        int elapsed   = (now - startTime) / 1000;
        int remaining = COLLECT_SECONDS - elapsed;
        if (remaining < 0) remaining = 0;
        showAcquiring(remaining, sampleCount);
        Serial.printf("  %ds — samples: %d/%d\n",
                      elapsed, sampleCount, TOTAL_SAMPLES);
      }

      // Read sensor
      readReg(REG_INTR_STATUS1);
      int avail = fifoAvailable();

      for (int s = 0; s < avail && sampleCount < TOTAL_SAMPLES; s++) {
        long irRaw = 0, redRaw = 0;
        if (!readFIFOSample(redRaw, irRaw)) continue;

        // Finger lost check
        if (irRaw < FINGER_THRESHOLD) {
          Serial.println("Finger lost during acquisition!");
          showError("Finger lost!\nKeep finger still");
          delay(3000);
          sampleCount    = 0;
          currentState   = STATE_IDLE;
          resetSignalState();
          showIdle();
          return;
        }

        // DC removal
        if (irDC  == 0) irDC  = irRaw;
        if (redDC == 0) redDC = redRaw;
        irDC  = ALPHA_IR  * irDC  + (1.0 - ALPHA_IR)  * irRaw;
        redDC = ALPHA_RED * redDC + (1.0 - ALPHA_RED)  * redRaw;

        // AC component (inverted so peaks go upward)
        float irAC  = -(irRaw  - irDC);
        float redAC = -(redRaw - redDC);

        // Smoothing
        float irSm  = smoothSignal(irSmooth,  irAC);
        float redSm = smoothSignal(redSmooth, redAC);
        smoothIndex++;

        // Store
        irBuffer[sampleCount]  = irSm;
        redBuffer[sampleCount] = redSm;
        sampleCount++;
      }

      yield();   // prevent watchdog reset
    }

    Serial.printf("Collection complete: %d samples\n", sampleCount);
    currentState = STATE_SEND;
    return;
  }

  // ── STATE: SEND ────────────────────────────────────────────────
  if (currentState == STATE_SEND) {
    showSending();
    Serial.println("Sending data to API...");

    // Reconnect WiFi if needed
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi lost — reconnecting...");
      connectWiFi();
    }

    // ── POST request — streamed directly to the socket ──────────
    int    httpCode = 0;
    String response;
    bool   sent = sendPredictRequest(httpCode, response);
    Serial.printf("HTTP code: %d (sent=%d)\n", httpCode, sent);

    if (sent && httpCode == 200) {
      Serial.println("Response: " + response);

      DynamicJsonDocument doc(1024);
      DeserializationError err = deserializeJson(doc, response);

      if (!err && doc["status"] == "success") {
        float  glucose = doc["glucose"].as<float>();
        String zone    = doc["zone"].as<String>();
        int    nSegs   = doc["n_segments_used"].as<int>();

        Serial.printf("Glucose: %.2f mg/dL | Zone: %s | Segs: %d\n",
                      glucose, zone.c_str(), nSegs);

        showResult(glucose, zone);

        // Hold result for 15 seconds
        delay(15000);

        // Return to idle
        showIdle();
        currentState = STATE_IDLE;

      } else {
        String msg = "Unknown error";
        if (!err && doc.containsKey("message")) {
          msg = doc["message"].as<String>();
        }
        Serial.println("API error: " + msg);
        showError(msg.substring(0, 40));
        delay(5000);
        showIdle();
        currentState = STATE_IDLE;
      }

    } else {
      Serial.printf("HTTP error: %d (sent=%d)\n", httpCode, sent);
      showError(sent ? ("HTTP Err: " + String(httpCode)) : "Connection\nfailed");
      delay(5000);
      showIdle();
      currentState = STATE_IDLE;
    }

    return;
  }
}
