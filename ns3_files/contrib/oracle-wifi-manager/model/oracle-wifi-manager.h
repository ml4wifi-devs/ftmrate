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
  WifiRemoteStation *DoCreateStation (void) const override;
  void DoReportRxOk (WifiRemoteStation *station, double rxSnr, WifiMode txMode) override;
  void DoReportAmpduTxStatus (WifiRemoteStation *station, uint16_t nSuccessfulMpdus,
                              uint16_t nFailedMpdus, double rxSnr, double dataSnr,
                              uint16_t dataChannelWidth, uint8_t dataNss) override;
  void DoReportRtsFailed (WifiRemoteStation *station) override;
  void DoReportDataFailed (WifiRemoteStation *station) override;
  void DoReportRtsOk (WifiRemoteStation *station, double ctsSnr, WifiMode ctsMode,
                      double rtsSnr) override;
  void DoReportDataOk (WifiRemoteStation *station, double ackSnr, WifiMode ackMode, double dataSnr,
                       uint16_t dataChannelWidth, uint8_t dataNss) override;
  void DoReportFinalRtsFailed (WifiRemoteStation *station) override;
  void DoReportFinalDataFailed (WifiRemoteStation *station) override;
  WifiTxVector DoGetDataTxVector (WifiRemoteStation *station) override;
  WifiTxVector DoGetRtsTxVector (WifiRemoteStation *station) override;

  WifiMode m_ctlMode;   // Wi-Fi mode for RTS frames
  double m_distance;    // current distance between STA and AP
};

} //namespace ns3

#endif /* ORACLE_WIFI_MANAGER_H */
