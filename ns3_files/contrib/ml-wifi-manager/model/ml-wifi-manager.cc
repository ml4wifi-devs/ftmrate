/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/string.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-tx-vector.h"
#include "ns3/wifi-utils.h"
#include "ml-wifi-manager.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MlWifiManager");

NS_OBJECT_ENSURE_REGISTERED (MlWifiManager);

TypeId
MlWifiManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MlWifiManager")
    .SetParent<WifiRemoteStationManager> ()
    .SetGroupName ("Wifi")
    .AddConstructor<MlWifiManager> ()
    .AddAttribute ("ControlMode", "The transmission mode to use for every RTS packet transmission.",
                   StringValue ("OfdmRate6Mbps"),
                   MakeWifiModeAccessor (&MlWifiManager::m_ctlMode),
                   MakeWifiModeChecker ())
    .AddAttribute ("Distance", "Current distance between STA and AP [m]",
                   DoubleValue (0.),
                   MakeDoubleAccessor (&MlWifiManager::m_distance),
                   MakeDoubleChecker<double_t> ())
    .AddAttribute ("Power", "Current transmission power [dBm]",
                   DoubleValue (16.0206),
                   MakeDoubleAccessor (&MlWifiManager::m_power),
                   MakeDoubleChecker<double_t> ())
  ;
  return tid;
}

MlWifiManager::MlWifiManager ()
{
  NS_LOG_FUNCTION (this);

  m_env = new Ns3AIRL<sEnv, sAct> (2333);
  m_env->SetCond (2, 0);
}

MlWifiManager::~MlWifiManager ()
{
  NS_LOG_FUNCTION (this);
}

WifiRemoteStation *
MlWifiManager::DoCreateStation (void) const
{
  NS_LOG_FUNCTION (this);

  MlWifiRemoteStation *st = new MlWifiRemoteStation ();
  st->last_sample = Simulator::Now ().GetSeconds ();
  st->m_mode = 0;

  auto env = m_env->EnvSetterCond ();
  env->time = Simulator::Now ().GetSeconds ();
  env->power = GetPhy ()->GetPowerDbm (GetDefaultTxPowerLevel ());
  env->mode = 0;
  env->type = 0;
  m_env->SetCompleted ();

  auto act = m_env->ActionGetterCond ();
  st->m_station_id = act->station_id;
  m_env->GetCompleted ();

  return st;
}

void
MlWifiManager::DoReportRxOk (WifiRemoteStation *station, double rxSnr, WifiMode txMode)
{
  NS_LOG_FUNCTION (this << station << rxSnr << txMode);
}

void
MlWifiManager::DoReportAmpduTxStatus (WifiRemoteStation *station, uint16_t nSuccessfulMpdus,
                                      uint16_t nFailedMpdus, double rxSnr, double dataSnr,
                                      uint16_t dataChannelWidth, uint8_t dataNss)
{
  SampleModeWrapper (station, nSuccessfulMpdus, nFailedMpdus, 2);
  NS_LOG_FUNCTION (this << station << nSuccessfulMpdus << nFailedMpdus << rxSnr << dataSnr
                        << dataChannelWidth << dataNss);
}

void
MlWifiManager::DoReportRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
MlWifiManager::DoReportDataFailed (WifiRemoteStation *station)
{
  SampleModeWrapper (station, 0, 1, 0);
  NS_LOG_FUNCTION (this << station);
}

void
MlWifiManager::DoReportRtsOk (WifiRemoteStation *station, double ctsSnr, WifiMode ctsMode,
                              double rtsSnr)
{
  NS_LOG_FUNCTION (this << station << ctsSnr << ctsMode << rtsSnr);
}

void
MlWifiManager::DoReportDataOk (WifiRemoteStation *station, double ackSnr, WifiMode ackMode,
                               double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss)
{
  SampleModeWrapper (station, 1, 0, 1);
  NS_LOG_FUNCTION (this << station << ackSnr << ackMode << dataSnr << dataChannelWidth << +dataNss);
}

void
MlWifiManager::DoReportFinalRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
MlWifiManager::DoReportFinalDataFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

WifiTxVector
MlWifiManager::DoGetDataTxVector (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);

  auto st = static_cast<MlWifiRemoteStation *> (station);

  WifiMode mode ("HeMcs" + std::to_string (st->m_mode));

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
MlWifiManager::DoGetRtsTxVector (WifiRemoteStation *station)
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

void
MlWifiManager::SampleMode(MlWifiRemoteStation *st, uint16_t nSuccessful, uint16_t nFailed, uint8_t report_source)
{
  auto env = m_env->EnvSetterCond ();
  env->power = m_power;
  env->time = Simulator::Now ().GetSeconds ();
  env->distance = m_distance;
  env->station_id = st->m_station_id;
  env->n_successful = nSuccessful;
  env->n_failed = nFailed;
  env->mode = st->m_mode;
  env->type = 1;
  env->report_source = report_source;
  m_env->SetCompleted ();

  auto act = m_env->ActionGetterCond ();
  if (act->station_id != st->m_station_id)
    {
      std::cout << "Env sid: " << st->m_station_id << " Act sid: " << act->station_id << std::endl;
      NS_ASSERT_MSG (
          act->station_id == st->m_station_id,
          "Error! Difference between station_id in ns3-ai action and remote station structures!");
    }
  st->m_mode = act->mode;
  m_env->GetCompleted ();

  st->last_sample = Simulator::Now ().GetSeconds ();
}

void MlWifiManager::SampleModeWrapper (WifiRemoteStation *station, uint16_t nSuccessful, uint16_t nFailed, uint8_t report_source)
{
  auto st = static_cast<MlWifiRemoteStation *> (station);

  #ifdef SAMPLE_INTERVAL
  if (Simulator::Now ().GetSeconds () - st->last_sample >= SAMPLE_INTERVAL)
    {
      SampleMode (st, nSuccessful, nFailed, report_source);
    }
  #else
    SampleMode (st, nSuccessful, nFailed, report_source);
  #endif
}

} //namespace ns3
