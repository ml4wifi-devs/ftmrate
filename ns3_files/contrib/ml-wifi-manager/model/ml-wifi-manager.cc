/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "ns3/ap-wifi-mac.h"
#include "ns3/core-module.h"
#include "ns3/ftm-error-model.h"
#include "ns3/ftm-header.h"
#include "ns3/ftm-manager.h"
#include "ns3/ftm-session.h"
#include "ns3/mac48-address.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-tx-vector.h"
#include "ns3/wifi-utils.h"
#include "ml-wifi-manager.h"

#define SAMPLE_INTERVAL 0.01
#define RTT_TO_DISTANCE 0.00015
#define FTM_RETRY_TIME 0.05
#define MAX_DISTANCE 1000.

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
    .AddAttribute ("ControlMode", "The transmission mode to use for every RTS packet transmission",
                   StringValue ("OfdmRate6Mbps"),
                   MakeWifiModeAccessor (&MlWifiManager::m_ctlMode),
                   MakeWifiModeChecker ())
    .AddAttribute ("Power", "Current transmission power [dBm]",
                   DoubleValue (16.0206),
                   MakeDoubleAccessor (&MlWifiManager::m_power),
                   MakeDoubleChecker<double_t> ())
    .AddAttribute ("WifiNetDevice", "WifiNetDevice of the device where the manager is installed",
                   PointerValue (),
                   MakePointerAccessor (&MlWifiManager::SetWifiNetDevice, &MlWifiManager::GetWifiNetDevice),
                   MakePointerChecker<WifiNetDevice> ())
    .AddAttribute ("ApAddress", "A MAC address of the access point",
                   Mac48AddressValue (Mac48Address ("ff:ff:ff:ff:ff:ff")),
                   MakeMac48AddressAccessor (&MlWifiManager::m_apAddr),
                   MakeMac48AddressChecker ())
  ;
  return tid;
}

MlWifiManager::MlWifiManager ()
{
  NS_LOG_FUNCTION (this);

  m_env = new Ns3AIRL<sEnv, sAct> (2333);
  m_env->SetCond (2, 0);

  m_device = 0;
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
  env->power = GetPhy ()->GetPowerDbm (GetDefaultTxPowerLevel ());
  env->time = Simulator::Now ().GetSeconds ();
  env->distance = 0.;
  env->mode = 0;
  env->type = 0;
  env->ftm_completed = false;
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
MlWifiManager::DoReportAmpduTxStatus (WifiRemoteStation *station, uint8_t nSuccessfulMpdus,
                                      uint8_t nFailedMpdus, double rxSnr, double dataSnr,
                                      uint16_t dataChannelWidth, uint8_t dataNss)
{
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

#ifdef SAMPLE_INTERVAL
  if (Simulator::Now ().GetSeconds () - st->last_sample >= SAMPLE_INTERVAL)
    {
      SampleMode (st);
    }
#else
  SampleMode (st);
#endif

  WifiMode mode ("HeMcs" + std::to_string (st->m_mode));

  return WifiTxVector (
      mode,
      GetDefaultTxPowerLevel (),
      GetPreambleForTransmission (mode.GetModulationClass (), GetShortPreambleEnabled (), UseGreenfieldForDestination (GetAddress (station))),
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
      GetPreambleForTransmission (m_ctlMode.GetModulationClass (), GetShortPreambleEnabled (), UseGreenfieldForDestination (GetAddress (station))),
      ConvertGuardIntervalToNanoSeconds (m_ctlMode, GetShortGuardIntervalSupported (station), NanoSeconds (GetGuardInterval (station))),
      1,
      1,
      0,
      GetChannelWidthForTransmission (m_ctlMode, GetChannelWidth (station)),
      GetAggregation (station));
}

void
MlWifiManager::SetWifiNetDevice(Ptr <WifiNetDevice> device)
{
  m_device = device;
}

Ptr<WifiNetDevice>
MlWifiManager::GetWifiNetDevice() const
{
  return m_device;
}

void
MlWifiManager::SampleMode(MlWifiRemoteStation *st)
{
  auto env = m_env->EnvSetterCond ();
  env->power = m_power;
  env->time = Simulator::Now ().GetSeconds ();
  env->distance = m_distance;
  env->station_id = st->m_station_id;
  env->mode = st->m_mode;
  env->type = 1;
  env->ftm_completed = m_ftmCompleted;
  m_env->SetCompleted ();

  auto act = m_env->ActionGetterCond ();
  if (act->station_id != st->m_station_id)
    {
      std::cout << "Env sid: " << st->m_station_id << " Act sid: " << act->station_id << std::endl;
      NS_ASSERT_MSG (act->station_id == st->m_station_id,
                     "Error! Difference between station_id in ns3-ai action and remote station structures!");
    }

  st->m_mode = act->mode;
  bool ftmRequest = act->ftm_request;

  m_env->GetCompleted ();

  if (ftmRequest)
    {
      Simulator::ScheduleNow (&MlWifiManager::FtmBurst, this);
    }

  st->last_sample = Simulator::Now ().GetSeconds ();
  m_ftmCompleted = false;
}

void
MlWifiManager::FtmBurst ()
{
  Ptr<RegularWifiMac> staMac = m_device->GetMac ()->GetObject<RegularWifiMac> ();
  Ptr<FtmSession> session = staMac->NewFtmSession (m_apAddr);

  if (session != NULL)
    {
      Ptr<WirelessSigStrFtmErrorModel> errorModel = CreateObject<WirelessSigStrFtmErrorModel> (RngSeedManager::GetRun ());
      errorModel->SetNode (m_device->GetNode ());

      session->SetFtmErrorModel (errorModel);
      session->SetSessionOverCallback (MakeCallback (&MlWifiManager::FtmSessionOver, this));
      session->SessionBegin ();
    }
}

void
MlWifiManager::FtmSessionOver (FtmSession session)
{
  if (session.GetIndividualRTT ().size () > 0 && session.GetMeanRTT () * RTT_TO_DISTANCE < MAX_DISTANCE)
    {
      m_distance = session.GetMeanRTT () * RTT_TO_DISTANCE;
      m_ftmCompleted = true;
    }
  else
    {
      Simulator::Schedule (Seconds (FTM_RETRY_TIME), &MlWifiManager::FtmBurst, this);
    }
}

} //namespace ns3
