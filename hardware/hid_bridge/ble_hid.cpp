#ifdef HID_TRANSPORT_BLE
#include "ble_hid.h"
#include "ble_link.h"
#include "hid_keys.h"
#include "board_support.h"
#include "ble_hid_target.h"
#include <NimBLEDevice.h>
#include <NimBLEHIDDevice.h>
#include "nimble/nimble/host/include/host/ble_hs.h"
#include "USBHIDKeyboard.h"
#include "tusb.h"

static NimBLEHIDDevice *hidDevice;
static NimBLECharacteristic *inputs[3];
static BleHidTarget target;
static unsigned lastGeneration;
static bool failed, releasePending;
static uint8_t mouseButtons, keys[8];
// Boot-mouse report: buttons, X, Y, wheel. No report ID, no pan byte.
static const size_t MOUSE_REPORT_SIZE = 4;
static uint32_t batteryAt, lastReportAt;
static volatile uint32_t reportsSent, reportsFailed, reportsWaited, reportMaxMs;  // ble_hid_diag

static void resetReports() {
  uint8_t zero[8] = {};
  inputs[0]->setValue(zero, MOUSE_REPORT_SIZE);
  inputs[1]->setValue(zero, 8);
  inputs[2]->setValue(zero, 2);
}

class HidSubscription : public NimBLECharacteristicCallbacks {
  void onSubscribe(NimBLECharacteristic *, NimBLEConnInfo &info, uint16_t sub) override {
    ble_hid_subscribe(info.getConnHandle(), sub & 1);
  }
};
static HidSubscription hidSubscription;

void ble_hid_begin(NimBLEServer *server) {
  auto *hid = new NimBLEHIDDevice(server);
  hidDevice = hid;
  // The mouse collection is emitted WITHOUT a report ID and with the boot-mouse
  // field layout (buttons, X, Y, wheel = 4 bytes). iOS only starts drawing a
  // pointer for an unnumbered pointing-device collection; with REPORT_ID(1) the
  // reports were accepted (hid=ok) but no cursor ever appeared.
  static uint8_t map[] = {
    0x05, 0x01, 0x09, 0x02, 0xA1, 0x01,        // Generic Desktop, Mouse, Application
    0x09, 0x01, 0xA1, 0x00,                    //   Pointer, Physical
    0x05, 0x09, 0x19, 0x01, 0x29, 0x03,        //     Buttons 1-3
    0x15, 0x00, 0x25, 0x01, 0x75, 0x01, 0x95, 0x03, 0x81, 0x02,
    0x95, 0x01, 0x75, 0x05, 0x81, 0x03,        //     padding
    0x05, 0x01, 0x09, 0x30, 0x09, 0x31,        //     X, Y
    0x15, 0x81, 0x25, 0x7F, 0x75, 0x08, 0x95, 0x02, 0x81, 0x06,
    0x09, 0x38, 0x15, 0x81, 0x25, 0x7F, 0x75, 0x08, 0x95, 0x01, 0x81, 0x06,  // Wheel
    0xC0, 0xC0,
    TUD_HID_REPORT_DESC_KEYBOARD(HID_REPORT_ID(2)),
    TUD_HID_REPORT_DESC_CONSUMER(HID_REPORT_ID(3))
  };
  hid->setReportMap(map, sizeof map);
  hid->setManufacturer("ESP32 iPhone Agent");
  hid->setPnp(2, 0x303a, 0x1001, 0x0101);
  hid->setHidInfo(0, 3);
  for (int i = 0; i < 3; i++) inputs[i] = hid->getInputReport(i + 1);
  resetReports();
  inputs[0]->setCallbacks(&hidSubscription);
  hid->getOutputReport(2)->setValue(uint8_t(0));
  // Let the board supply the battery level; a board without a battery reports zero.
  hid->setBatteryLevel(board_battery_percent());
  batteryAt = millis();
}

void ble_hid_disconnected(uint16_t handle) {
  target.disconnect(handle);
}

uint32_t ble_hid_session() { return target.session(); }

void ble_hid_claim(uint16_t handle, bool subscribed) {
  if (target.subscribe(handle, ble_hid_trusted(handle), subscribed)) resetReports();
}

bool ble_hid_allows(int handle) {
  return handle >= 0 && target.allows(handle, ble_hid_trusted(handle));
}

bool ble_hid_ready() {
  return ble_hid_trusted(BleHidTarget::peer(target.session()));
}

bool ble_hid_command_begin(uint32_t session) {
  if (session != target.session()) return false;
  failed = false;
  if (lastGeneration != session) {
    mouseButtons = 0;
    memset(keys, 0, sizeof keys);
    releasePending = false;
    lastGeneration = session;
  }
  return true;
}

bool ble_hid_command_ok() {
  // A press may already have reached the phone. Retry its release in the main loop.
  if (failed) releasePending = true;
  return !failed;
}

static void send(int input, const void *data, size_t size) {
  if (failed) return;
  // A notification can also fail during timer expiry or disconnect cleanup, outside a command.
  int handle = BleHidTarget::peer(lastGeneration);
  uint32_t start = millis();
  for (;;) {
    if (handle < 0 || target.session() != lastGeneration || !ble_hid_ready()) {
      failed = releasePending = true;
      reportsFailed++;
      return;
    }
    // As in the video path, retry only when the host runs out of buffers.
    os_mbuf *packet = ble_hs_mbuf_from_flat(data, size);
    int rc = packet ? ble_gatts_notify_custom(handle, inputs[input]->getHandle(), packet) : BLE_HS_ENOMEM;
    if (rc == 0) {
      lastReportAt = millis();
      uint32_t took = lastReportAt - start;
      reportsSent++;
      if (took) reportsWaited++;
      if (took > reportMaxMs) reportMaxMs = took;
      return;
    }
    if (rc != BLE_HS_ENOMEM || millis() - start >= 100) {
      failed = releasePending = true;
      reportsFailed++;
#if DEBUG_SERIAL
      Serial.printf("BLE HID report failed: rc=%d\n", rc);
#endif
      return;
    }
    vTaskDelay(1);
  }
}

void ble_hid_diag(uint32_t *sent, uint32_t *failedReports, uint32_t *waited, uint32_t *maxMs) {
  *sent = reportsSent;
  *failedReports = reportsFailed;
  *waited = reportsWaited;
  *maxMs = reportMaxMs;
  reportMaxMs = 0;
}

void ble_hid_poll(bool timedHold) {
  uint32_t now = millis();
  if (hidDevice && now - batteryAt >= 30000) {
    hidDevice->setBatteryLevel(board_battery_percent(), true);
    batteryAt = now;
  }
  // An explicit press_for has its own deadline. Unbounded idle presses still expire after five seconds.
  if (ble_hid_ready() && (releasePending || (!timedHold && mouseButtons && now - lastReportAt >= 5000))) {
    if (!ble_hid_command_begin(lastGeneration)) return;
    uint8_t zero[8] = {};
    send(0, zero, MOUSE_REPORT_SIZE);
    send(1, zero, 8);
    send(2, zero, 2);
    if (!failed) {
#if DEBUG_SERIAL
      if (releasePending) Serial.println("BLE HID inputs released after command failure");
#endif
      mouseButtons = 0;
      memset(keys, 0, sizeof keys);
      releasePending = false;
    } else {
      releasePending = true;
    }
  }
}

void BleHidMouse::move(int8_t x, int8_t y, int8_t wheel, int8_t pan) {
  (void)pan;
  int8_t report[MOUSE_REPORT_SIZE] = {static_cast<int8_t>(mouseButtons), x, y, wheel};
  send(0, report, sizeof report);
}
void BleHidMouse::press(uint8_t buttons) { mouseButtons |= buttons; move(0, 0); }
void BleHidMouse::release(uint8_t buttons) { mouseButtons &= ~buttons; move(0, 0); }

void BleHidKeyboard::press(uint8_t key) {
  if (!hid_key_press(keys, key, KeyboardLayout_en_US)) { failed = true; return; }
  send(1, keys, sizeof keys);
}
void BleHidKeyboard::releaseAll() { memset(keys, 0, sizeof keys); send(1, keys, sizeof keys); }
size_t BleHidKeyboard::write(uint8_t value) {
  if (value >= 128) { failed = true; return 0; }
  press(value);
  if (failed) return 0;  // the key-down never reached the phone: not typed
  delay(8);
  releaseAll();  // if only this fails, the character is in; ble_hid_poll releases the key
  delay(8);
  return 1;
}

void BleHidConsumer::press(uint16_t usage) {
  uint8_t report[] = {static_cast<uint8_t>(usage), static_cast<uint8_t>(usage >> 8)};
  send(2, report, sizeof report);
}
void BleHidConsumer::release() { press(0); }
#endif
