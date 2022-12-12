#include <chrono>
#include <filesystem>
#include <map>
#include <string>

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-helper.h"
#include "ns3/mobility-model.h"
#include "ns3/node-container.h"
#include "ns3/node-list.h"
#include "ns3/ssid.h"
#include "ns3/wifi-net-device.h"
#include "ns3/yans-wifi-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("success-probability");

/***** Functions declarations *****/

void InstallTrafficGenerator (Ptr<ns3::Node> fromNode, Ptr<ns3::Node> toNode, uint32_t port,
                              DataRate offeredLoad, uint32_t packetSize, double warmupTime,
                              double simulationTime, double fuzzTime);
void PhyRxOkCallback (Ptr<const Packet> packet, double snr, WifiMode mode, WifiPreamble preamble);
void PhyTxCallback (Ptr<const Packet> packet, WifiMode mode, WifiPreamble preamble, uint8_t txPower);
void ResetCounters ();

/***** Global variables and constants *****/

#define DEFAULT_NOISE_VALUE (-93.966)

uint64_t totalFrames;
uint64_t successfulFrames;

/***** Main with scenario definition *****/

int
main (int argc, char *argv[])
{
  // Initialize default simulation parameters
  double fuzzTime = 5.;
  double warmupTime = 10.;
  double simulationTime = 500.;

  std::string csvPath = "results.csv";

  bool ampdu = true;
  uint32_t packetSize = 1500;
  uint32_t dataRate = 125;
  uint32_t channelWidth = 20;
  uint32_t minGI = 3200;

  std::string lossModel = "Nakagami";
  double snr = 30.;
  uint32_t mode = 0;

  // Parse command line arguments
  CommandLine cmd;
  cmd.AddValue ("ampdu", "Use AMPDU (boolean flag)", ampdu);
  cmd.AddValue ("channelWidth", "Channel width (MHz)", channelWidth);
  cmd.AddValue ("csvPath", "Path to output CSV file", csvPath);
  cmd.AddValue ("dataRate", "Traffic generator data rate (Mb/s)", dataRate);
  cmd.AddValue ("fuzzTime", "Maximum fuzz value (s)", fuzzTime);
  cmd.AddValue ("lossModel", "Propagation loss model (LogDistance, Nakagami)", lossModel);
  cmd.AddValue ("minGI", "Shortest guard interval (ns)", minGI);
  cmd.AddValue ("mode", "Modulation and coding scheme", mode);
  cmd.AddValue ("packetSize", "Packets size (B)", packetSize);
  cmd.AddValue ("simulationTime", "Duration of simulation (s)", simulationTime);
  cmd.AddValue ("snr", "SNR value (dBm)", snr);
  cmd.AddValue ("warmupTime", "Duration of warmup stage (s)", warmupTime);
  cmd.Parse (argc, argv);

  if (mode > 11)
    {
      std::cerr << "Selected incorrect mode!";
      return 1;
    }

  // Print simulation settings to screen
  std::cout << std::endl
            << "Simulating an IEEE 802.11ax devices with the following settings:" << std::endl
            << "- frequency band: 5 GHz" << std::endl
            << "- max data rate: " << dataRate << " Mb/s" << std::endl
            << "- channel width: " << channelWidth << " Mhz" << std::endl
            << "- shortest guard interval: " << minGI << " ns" << std::endl
            << "- packets size: " << packetSize << " B" << std::endl
            << "- AMPDU: " << ampdu << std::endl
            << "- rate adaptation manager: ns3::ConstantRateWifiManager" << std::endl
            << "- mode: " << mode << std::endl
            << "- number of stations: 1" << std::endl
            << "- simulation time: " << simulationTime << " s" << std::endl
            << "- warmup time: " << warmupTime << " s" << std::endl
            << "- max fuzz time: " << fuzzTime << " s" << std::endl
            << "- loss model: " << lossModel << std::endl
            << "- SNR: " << snr << " dBm" << std::endl
            << "- mobility model: Distance" << std::endl
            << std::endl;

  // Create AP and stations
  NodeContainer wifiApNode (1);
  NodeContainer wifiStaNode (1);

  // Configure mobility
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);
  mobility.Install (wifiStaNode);

  // Configure wireless channel
  YansWifiPhyHelper phy;
  YansWifiChannelHelper channelHelper = YansWifiChannelHelper::Default ();

  // Set FixedRssLossModel to use const SNR value
  phy.Set ("RxGain", DoubleValue (0));
  channelHelper.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  channelHelper.AddPropagationLoss ("ns3::FixedRssLossModel", "Rss",
                                    DoubleValue (snr + DEFAULT_NOISE_VALUE));

  if (lossModel == "Nakagami")
    {
      // Add Nakagami fading
      channelHelper.AddPropagationLoss ("ns3::NakagamiPropagationLossModel");
    }
  else if (lossModel != "LogDistance")
    {
      std::cerr << "Selected incorrect loss model!";
      return 1;
    }
  phy.SetChannel (channelHelper.Create ());

  // Configure MAC layer
  WifiMacHelper mac;
  WifiHelper wifi;

  wifi.SetStandard (WIFI_STANDARD_80211ax);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue ("HeMcs" + std::to_string (mode)),
                                "ControlMode", StringValue ("HeMcs" + std::to_string (mode)));

  //Set SSID
  Ssid ssid = Ssid ("ns3-80211ax");
  mac.SetType ("ns3::StaWifiMac", "Ssid", SsidValue (ssid));

  // Create and configure Wi-Fi interfaces
  NetDeviceContainer staDevice;
  staDevice = wifi.Install (phy, mac, wifiStaNode);

  mac.SetType ("ns3::ApWifiMac", "Ssid", SsidValue (ssid));

  NetDeviceContainer apDevice;
  apDevice = wifi.Install (phy, mac, wifiApNode);

  // Manage AMPDU aggregation
  if (!ampdu)
    {
      Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/BE_MaxAmpduSize",
                   UintegerValue (0));
    }

  // Set channel width and shortest GI
  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/ChannelSettings",
               StringValue ("{0, " + std::to_string (channelWidth) + ", BAND_5GHZ, 0}"));

  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HeConfiguration/GuardInterval",
               TimeValue (NanoSeconds (minGI)));

  // Install an Internet stack
  InternetStackHelper stack;
  stack.Install (wifiApNode);
  stack.Install (wifiStaNode);

  // Configure IP addressing
  Ipv4AddressHelper address ("192.168.1.0", "255.255.255.0");
  Ipv4InterfaceContainer staNodeInterface = address.Assign (staDevice);
  Ipv4InterfaceContainer apNodeInterface = address.Assign (apDevice);

  // Configure applications
  DataRate applicationDataRate = DataRate (dataRate * 1e6);
  uint32_t portNumber = 9;

  InstallTrafficGenerator (wifiStaNode.Get (0), wifiApNode.Get (0), portNumber,
                           applicationDataRate, packetSize, simulationTime, warmupTime, fuzzTime);

  // Setup frames counting
  Simulator::Schedule (Seconds (warmupTime), &ResetCounters);
  
  uint32_t apNode = wifiApNode.Get (0)->GetId ();
  uint32_t staNode = wifiStaNode.Get (0)->GetId ();
  
  Config::ConnectWithoutContext ("/NodeList/" + std::to_string (apNode) +
                                     "/DeviceList/*/$ns3::WifiNetDevice/Phy/State/RxOk",
                                 MakeCallback (PhyRxOkCallback));

  Config::ConnectWithoutContext ("/NodeList/" + std::to_string (staNode) +
                                     "/DeviceList/*/$ns3::WifiNetDevice/Phy/State/Tx",
                                 MakeCallback (PhyTxCallback));

  // Define simulation stop time
  Simulator::Stop (Seconds (warmupTime + simulationTime));

  // Record start time
  std::cout << "Starting simulation..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now ();

  // Run the simulation!
  Simulator::Run ();

  // Record stop time and count duration
  auto finish = std::chrono::high_resolution_clock::now ();
  std::chrono::duration<double> elapsed = finish - start;

  std::cout << "Done!" << std::endl
            << "Elapsed time: " << elapsed.count () << " s" << std::endl
            << std::endl;

  // Print results
  std::cout << std::endl
            << "Results: " << std::endl
            << "Total number of frames: " << totalFrames << std::endl
            << "Number of successfully transmitted frames: " << successfulFrames << std::endl
            << std::endl;

  // Gather results in CSV format
  std::ostringstream csvOutput;
  csvOutput << mode << ',' << snr << ',' << totalFrames << ',' << successfulFrames << std::endl;

  // Print results to std output
  std::cout << "mode,snr,n,k"
            << std::endl
            << csvOutput.str ();

  // Print results to file
  std::ofstream outputFile (csvPath);
  outputFile << csvOutput.str ();
  std::cout << std::endl << "Simulation data saved to: " << csvPath << std::endl << std::endl;

  //Clean-up
  Simulator::Destroy ();

  return 0;
}

/***** Function definitions *****/

void
InstallTrafficGenerator (Ptr<ns3::Node> fromNode, Ptr<ns3::Node> toNode, uint32_t port,
                         DataRate offeredLoad, uint32_t packetSize, double warmupTime,
                         double simulationTime, double fuzzTime)
{
  // Get sink address
  Ptr<Ipv4> ipv4 = toNode->GetObject<Ipv4> ();
  Ipv4Address addr = ipv4->GetAddress (1, 0).GetLocal ();

  // Define type of service
  uint8_t tosValue = 0x70; //AC_BE

  // Add random fuzz to app start time
  Ptr<UniformRandomVariable> fuzz = CreateObject<UniformRandomVariable> ();
  fuzz->SetAttribute ("Min", DoubleValue (0.));
  fuzz->SetAttribute ("Max", DoubleValue (fuzzTime));
  double applicationsStart = fuzz->GetValue ();

  // Configure source and sink
  InetSocketAddress sinkSocket (addr, port);
  sinkSocket.SetTos (tosValue);
  PacketSinkHelper packetSinkHelper ("ns3::UdpSocketFactory", sinkSocket);

  OnOffHelper onOffHelper ("ns3::UdpSocketFactory", sinkSocket);
  onOffHelper.SetConstantRate (offeredLoad, packetSize);

  // Configure applications
  ApplicationContainer sinkApplications (packetSinkHelper.Install (toNode));
  ApplicationContainer sourceApplications (onOffHelper.Install (fromNode));

  sinkApplications.Start (Seconds (applicationsStart));
  sinkApplications.Stop (Seconds (warmupTime + simulationTime));
  sourceApplications.Start (Seconds (applicationsStart));
  sourceApplications.Stop (Seconds (warmupTime + simulationTime));
}

void
PhyRxOkCallback (Ptr<const Packet> packet, double snr, WifiMode mode, WifiPreamble preamble)
{
  if (preamble == ns3::WIFI_PREAMBLE_HE_SU)
    {
      successfulFrames++;
    }
}

void
PhyTxCallback (Ptr<const Packet> packet, WifiMode mode, WifiPreamble preamble, uint8_t txPower)
{
  if (preamble == ns3::WIFI_PREAMBLE_HE_SU)
    {
      totalFrames++;
    }
}

void
ResetCounters ()
{
  totalFrames = 0;
  successfulFrames = 0;
}
