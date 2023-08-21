/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#ifndef ML_WIFI_MANAGER_H
#define ML_WIFI_MANAGER_H

#include "ns3/ftm-session.h"
#include "ns3/ns3-ai-module.h"
#include "ns3/wifi-remote-station-manager.h"

namespace ns3 {

// ns3-ai structures
struct sEnv
{
  double power;
  double time;
  double distance;
  uint32_t station_id;
  uint8_t mode;
  uint8_t type;
  bool ftm_completed;
} Packed;

struct sAct
{
  uint32_t station_id;
  uint8_t mode;
  bool ftm_request;
} Packed;

struct MlWifiRemoteStation : public WifiRemoteStation
{
  double last_sample;
  uint32_t m_station_id;
  uint8_t m_mode;
};

class MlWifiManager : public WifiRemoteStationManager
{
public:
  static TypeId GetTypeId (void);
  MlWifiManager ();
  virtual ~MlWifiManager ();

private:
  WifiRemoteStation *DoCreateStation (void) const;
  void DoReportRxOk (WifiRemoteStation *station, double rxSnr, WifiMode txMode);
  void DoReportAmpduTxStatus (WifiRemoteStation *station, uint16_t nSuccessfulMpdus,
                              uint16_t nFailedMpdus, double rxSnr, double dataSnr,
                              uint16_t dataChannelWidth, uint8_t dataNss);
  void DoReportRtsFailed (WifiRemoteStation *station);
  void DoReportDataFailed (WifiRemoteStation *station);
  void DoReportRtsOk (WifiRemoteStation *station, double ctsSnr, WifiMode ctsMode,
                      double rtsSnr);
  void DoReportDataOk (WifiRemoteStation *station, double ackSnr, WifiMode ackMode, double dataSnr,
                       uint16_t dataChannelWidth, uint8_t dataNss);
  void DoReportFinalRtsFailed (WifiRemoteStation *station);
  void DoReportFinalDataFailed (WifiRemoteStation *station);
  WifiTxVector DoGetDataTxVector (WifiRemoteStation *station);
  WifiTxVector DoGetRtsTxVector (WifiRemoteStation *station);

  void SetWifiNetDevice (Ptr<WifiNetDevice> device);
  Ptr<WifiNetDevice> GetWifiNetDevice (void) const;

  void SampleMode(MlWifiRemoteStation *station);
  void FtmBurst ();
  void FtmSessionOver (FtmSession session);

  WifiMode m_ctlMode;   // Wi-Fi mode for RTS frames
  double m_power;       // current tx power
  double m_distance;    // current distance between STA and AP
  bool m_ftmCompleted;  // has the FTM measurement been completed
  bool m_idealDistance; // pass the ideal distance to the module

  Ptr<WifiNetDevice> m_device;  // device where the manager is installed
  Mac48Address m_apAddr;        // MAC address of the AP

  Ns3AIRL<sEnv, sAct> *m_env;
};

} //namespace ns3

#endif /* ML_WIFI_MANAGER_H */
