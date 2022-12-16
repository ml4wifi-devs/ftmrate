/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/string.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-tx-vector.h"
#include "ns3/wifi-utils.h"
#include "oracle-wifi-manager.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("OracleWifiManager");

NS_OBJECT_ENSURE_REGISTERED (OracleWifiManager);

double lookupTable[] = {7.85, 7.91, 15.79, 17.49,
                        24.76, 28.21, 32.94, 84.21,
                        84.21, 4106., 4106., std::numeric_limits<double>::infinity()};

TypeId
OracleWifiManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::OracleWifiManager")
    .SetParent<WifiRemoteStationManager> ()
    .SetGroupName ("Wifi")
    .AddConstructor<OracleWifiManager> ()
    .AddAttribute ("ControlMode", "The transmission mode to use for every RTS packet transmission.",
                   StringValue ("OfdmRate6Mbps"),
                   MakeWifiModeAccessor (&OracleWifiManager::m_ctlMode),
                   MakeWifiModeChecker ())
    .AddAttribute ("Distance", "Current distance between STA and AP [m]",
                   DoubleValue (0.),
                   MakeDoubleAccessor (&OracleWifiManager::m_distance),
                   MakeDoubleChecker<double_t> ())
  ;
  return tid;
}

OracleWifiManager::OracleWifiManager ()
{
  NS_LOG_FUNCTION (this);
}

OracleWifiManager::~OracleWifiManager ()
{
  NS_LOG_FUNCTION (this);
}

WifiRemoteStation *
OracleWifiManager::DoCreateStation (void) const
{
  NS_LOG_FUNCTION (this);
  return new WifiRemoteStation ();
}

void
OracleWifiManager::DoReportRxOk (WifiRemoteStation *station, double rxSnr, WifiMode txMode)
{
  NS_LOG_FUNCTION (this << station << rxSnr << txMode);
}

void
OracleWifiManager::DoReportAmpduTxStatus (WifiRemoteStation *station, uint16_t nSuccessfulMpdus,
                                      uint16_t nFailedMpdus, double rxSnr, double dataSnr,
                                      uint16_t dataChannelWidth, uint8_t dataNss)
{
  NS_LOG_FUNCTION (this << station << nSuccessfulMpdus << nFailedMpdus << rxSnr << dataSnr
                        << dataChannelWidth << dataNss);
}

void
OracleWifiManager::DoReportRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
OracleWifiManager::DoReportDataFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
OracleWifiManager::DoReportRtsOk (WifiRemoteStation *station, double ctsSnr, WifiMode ctsMode,
                              double rtsSnr)
{
  NS_LOG_FUNCTION (this << station << ctsSnr << ctsMode << rtsSnr);
}

void
OracleWifiManager::DoReportDataOk (WifiRemoteStation *station, double ackSnr, WifiMode ackMode,
                               double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss)
{
  NS_LOG_FUNCTION (this << station << ackSnr << ackMode << dataSnr << dataChannelWidth << +dataNss);
}

void
OracleWifiManager::DoReportFinalRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
OracleWifiManager::DoReportFinalDataFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

WifiTxVector
OracleWifiManager::DoGetDataTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);

  uint8_t modeIdx = 0;
  while (modeIdx < 11 && lookupTable[modeIdx] < m_distance)
    {
      modeIdx++;
    }

  WifiMode mode ("HeMcs" + std::to_string (11 - modeIdx));

  return WifiTxVector (
      mode,
      GetDefaultTxPowerLevel (),
      GetPreambleForTransmission (mode.GetModulationClass (), GetShortPreambleEnabled ()),
      ConvertGuardIntervalToNanoSeconds (mode, GetShortGuardIntervalSupported (st), NanoSeconds (GetGuardInterval (st))),
      GetNumberOfAntennas (),
      1,
      0,
      GetChannelWidthForTransmission (mode, GetChannelWidth (st)),
      GetAggregation (st));
}

WifiTxVector
OracleWifiManager::DoGetRtsTxVector (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);

  return WifiTxVector (
      m_ctlMode,
      GetDefaultTxPowerLevel (),
      GetPreambleForTransmission (m_ctlMode.GetModulationClass (), GetShortPreambleEnabled ()),
      ConvertGuardIntervalToNanoSeconds (m_ctlMode, GetShortGuardIntervalSupported (station), NanoSeconds (GetGuardInterval (station))),
      1,
      1,
      0,
      GetChannelWidthForTransmission (m_ctlMode, GetChannelWidth (station)),
      GetAggregation (station));
}

} //namespace ns3
