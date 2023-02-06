/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#ifndef ORACLE_WIFI_MANAGER_H
#define ORACLE_WIFI_MANAGER_H

#include "ns3/wifi-remote-station-manager.h"

namespace ns3 {

class OracleWifiManager : public WifiRemoteStationManager
{
public:
  static TypeId GetTypeId (void);
  OracleWifiManager ();
  virtual ~OracleWifiManager ();

private:
  WifiRemoteStation *DoCreateStation (void) const;
  void DoReportRxOk (WifiRemoteStation *station, double rxSnr, WifiMode txMode);
  void DoReportAmpduTxStatus (WifiRemoteStation *station, uint8_t nSuccessfulMpdus,
                              uint8_t nFailedMpdus, double rxSnr, double dataSnr,
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

  static double DistanceToSnr (double distance, double power);
  uint8_t GetBestMcs ();

  WifiMode m_ctlMode;   // Wi-Fi mode for RTS frames
  double m_distance;    // current distance between STA and AP
  double m_power;       // current tx power
};

} //namespace ns3

#endif /* ORACLE_WIFI_MANAGER_H */
