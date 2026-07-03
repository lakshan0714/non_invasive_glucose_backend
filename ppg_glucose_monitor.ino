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
#include <ArduinoJson.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_GFX.h>
#include <esp_heap_caps.h>

// ── WiFi credentials ──────────────────────────────────────────────
const char* WIFI_SSID     = "chiki chiki";       // ← change
const char* WIFI_PASSWORD = "thuwa567891011";   // ← change

// ── API endpoints ─────────────────────────────────────────────────
const char* API_URL    = "https://non-invasive-glucose-backend.onrender.com/predict";
const char* HEALTH_URL = "https://non-invasive-glucose-backend.onrender.com/health";

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

// ── JSON payload buffer (built once at boot, reused every send) ──
// Worst case per float: "-262143.1234," = 14 bytes. 2 arrays * TOTAL_SAMPLES.
const size_t PAYLOAD_BUF_SIZE = (size_t)TOTAL_SAMPLES * 2 * 14 + 256;
char*  payloadBuf = nullptr;

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

  // Allocate JSON payload buffer ONCE, here, while heap is least
  // fragmented. Prefer PSRAM if the board has it; fall back to
  // internal heap otherwise. Building this buffer with repeated
  // Arduino String concatenation instead (as before) caused heap
  // fragmentation/OOM mid-request, which silently emptied the String
  // and sent an empty POST body (server saw 422 "Field required").
  Serial.printf("Free heap before payload buffer alloc: %u bytes\n", ESP.getFreeHeap());
  Serial.printf("Requesting payload buffer: %u bytes\n", (unsigned)PAYLOAD_BUF_SIZE);

  payloadBuf = (char*) heap_caps_malloc(PAYLOAD_BUF_SIZE, MALLOC_CAP_SPIRAM);
  if (payloadBuf) {
    Serial.println("Payload buffer allocated in PSRAM");
  } else {
    payloadBuf = (char*) malloc(PAYLOAD_BUF_SIZE);
    if (payloadBuf) Serial.println("Payload buffer allocated in internal heap");
  }

  if (!payloadBuf) {
    Serial.println("Payload buffer allocation failed!");
    showError("Memory error!\nRestart device");
    while (true);
  }
  Serial.printf("Free heap after payload buffer alloc: %u bytes\n", ESP.getFreeHeap());

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

    HTTPClient http;
    http.begin(API_URL);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(120000);   // 2 min timeout

    // ── Build JSON into the pre-allocated buffer ─────────────────
    // Writing directly into one fixed buffer (instead of growing an
    // Arduino String via repeated +=) avoids the heap fragmentation/OOM
    // that was silently truncating the payload to empty.
    Serial.printf("Free heap before JSON build: %u bytes\n", ESP.getFreeHeap());

    size_t pos = 0;
    size_t cap = PAYLOAD_BUF_SIZE;
    bool   overflow = false;

    #define APPEND(...) do { \
      int _n = snprintf(payloadBuf + pos, cap - pos, __VA_ARGS__); \
      if (_n < 0 || (size_t)_n >= cap - pos) { overflow = true; } \
      else { pos += (size_t)_n; } \
    } while (0)

    APPEND("{\"age\":%.1f,\"ir\":[", PATIENT_AGE);

    for (int i = 0; i < TOTAL_SAMPLES && !overflow; i++) {
      APPEND(i < TOTAL_SAMPLES - 1 ? "%.4f," : "%.4f", irBuffer[i]);
      if (i % 500 == 0) {
        yield();
        Serial.printf("Building JSON IR: %d/%d\n", i, TOTAL_SAMPLES);
      }
    }
    if (!overflow) APPEND("],\"red\":[");

    for (int i = 0; i < TOTAL_SAMPLES && !overflow; i++) {
      APPEND(i < TOTAL_SAMPLES - 1 ? "%.4f," : "%.4f", redBuffer[i]);
      if (i % 500 == 0) yield();
    }
    if (!overflow) APPEND("]}");

    #undef APPEND

    if (overflow) {
      Serial.println("Payload buffer overflow — aborting send!");
      showError("Payload too\nlarge, restart");
      delay(5000);
      showIdle();
      currentState = STATE_IDLE;
      return;
    }

    Serial.printf("Payload size: %u bytes\n", (unsigned)pos);
    Serial.printf("Free heap after JSON build: %u bytes\n", ESP.getFreeHeap());

    // ── POST request ────────────────────────────────────────────
    int httpCode = http.POST((uint8_t*)payloadBuf, pos);
    Serial.printf("HTTP code: %d\n", httpCode);

    if (httpCode == 200) {
      String response = http.getString();
      Serial.println("Response: " + response);

      DynamicJsonDocument doc(1024);
      DeserializationError err = deserializeJson(doc, response);

      if (!err && doc["status"] == "success") {
        float  glucose = doc["glucose"].as<float>();
        String zone    = doc["zone"].as<String>();
        int    nSegs   = doc["n_segments_used"].as<int>();

        Serial.printf("Glucose: %.2f mg/dL | Zone: %s | Segs: %d\n",
                      glucose, zone.c_str(), nSegs);

        http.end();
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
        http.end();
        showError(msg.substring(0, 40));
        delay(5000);
        showIdle();
        currentState = STATE_IDLE;
      }

    } else {
      Serial.printf("HTTP error: %d\n", httpCode);
      http.end();
      showError("HTTP Err: " + String(httpCode));
      delay(5000);
      showIdle();
      currentState = STATE_IDLE;
    }

    return;
  }
}
