/*

This file is part of Web Research Manager.

Web Research Manager is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Web Research Manager is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Web Research Manager.If not, see<https://www.gnu.org/licenses/>. 

*/


using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using System.Diagnostics;
using HtmlAgilityPack;
using ZedGraph;
using System.IO;
using System.Data.SQLite;
using System.Reflection;
using Microsoft.Web.WebView2.Core;
using Newtonsoft.Json;
using System.Net;
using Microsoft.Web.WebView2.WinForms;
using Python.Runtime;

namespace WebResearchManager
{
    public partial class FrmMain : Form
    {
        public FrmMain()
        {
            InitializeComponent();
        }

        private FrameWebView2 c;
        private FrameWebView2 editorInput;

        //TODO: this should be read in from a file
        private string pythonPath;
        private string debugClickbaitDetector = "\\detectors\\clickbait\\nvsclickbait.py";
        private string debugSatireDetector = "\\detectors\\satire\\nvssatire.py";
        private string debugFalsificationDetector = "\\detectors\\falsifications\\nvsfalsifications.py";

        private Process clickbaitDetector;
        private Process satireDetector;
        private Process falsificationDetector;

        private HtmlAgilityPack.HtmlDocument doc;
        private List<string> clickbaitResults = new List<string>();

        private FrmStatistics frmStats;
        private FrmErrors frmErrors;
        private FrameClickbait clickbaitPanel;

        //clickbait

        //need OS and Dill modules as imports for my clickbait detector
        private dynamic pyOs;
        private dynamic pyDill;

        //clickbaitml.py is the file with all the detector stuff
        private dynamic pyClickbaitMl;

        //nvsclickbait.py basically just has the opendill(..) function ...
        private dynamic pyNvsClickbait;

        //opendill(...) puts it in the detector dump variable below
        private dynamic pyClickbaitDetectorDUMP;

        //this actually loads the dump file... see line 203
        private dynamic pyClickbaitDetector;

        //MSc students ... add your detectors here, at line 197, and test them with some input <a> link captions at line 847
        //this may not be the best input for your detector (<p> or <div> text might be better) but that's all that works well right now
        //note that you dont have to do GUI stuff. You are free to do that though if you have time, look at the Satire or Falsifications panels and code

        //Make sure supervised models load and can process text.
        //ALSO make sure the code for the GPU-based deep learning stuff works, but you might need to comment it out since 6 of them could fill up a desktop GPU pretty quick.

        //public health


        //health relatedness


        //humor


        //hyperpartisian
        private dynamic pyHyperpartisanMl;

        //opendill(...) puts it in the detector dump variable below
        private dynamic pyHyperpartisanDetectorDUMP;

        //this actually loads the dump file... see line 203
        private dynamic pyHyperpartisanDetector;

        //political bias


        //fake news


        private void FrmMain_Load(object sender, EventArgs e)
        {
            // Get the version of the executing assembly (that is, this assembly).
            Assembly assem = Assembly.GetEntryAssembly();
            AssemblyName assemName = assem.GetName();
            Version ver = assemName.Version;

            this.Text += ver.ToString();

            frmStats = new FrmStatistics(this);
            frmErrors = new FrmErrors();

            cboSatireDetectorAcc.SelectedIndex = 0;
            cboFalsificationDetectorAcc.SelectedIndex = 0;

            //only call this once for pythonnet at beginning
            //you must set this to your python dll on your PC for now otherwise this program won't work correctly
            Runtime.PythonDLL = "C:\\Users\\DHANASHREE\\AppData\\Local\\Programs\\Python\\Python311\\python311.dll";
            PythonEngine.Initialize();

            //inputWebBrowser.DocumentText = "<HTML><BODY contentEditable='true'></BODY></HTML>";

            // //a | //span | //h1 | //h2 | //h3 | //h4 | //h5
            string[] clickbaitDefaultTags = { "a", "span", "h1", "h2", "h3", "h4", "h5", "yt-formatted-string" };
            clickbaitTagWnd = new FrmSelectHtmlTags(clickbaitDefaultTags);

            string[] satFalsDefaultTags = { "p" };

            // //p
            satireTagWnd = new FrmSelectHtmlTags(satFalsDefaultTags);

            // //p
            falsificationTagWnd = new FrmSelectHtmlTags(satFalsDefaultTags);

            //adding the clickbait frame here because it crashes in the designer since this is a 64-bit app
            clickbaitPanel = new FrameClickbait();
            PanelMain.Controls.Add(clickbaitPanel);
            clickbaitPanel.Dock = DockStyle.Fill;

            ListView clickbaitView = (ListView)clickbaitPanel.Controls.Find("LstViewClickbait", false)[0];

            //left click shows stats, right click shows user scores
            clickbaitView.MouseClick += ClickbaitView_MouseClick;
            clickbaitView.MouseDoubleClick += ClickbaitView_MouseDoubleClick;

            //load all paths -- TODO -- just python to fix Vicki's install
            pythonPath = File.ReadAllText("py.path");
            TxtHomepageURL.Text = File.ReadAllText("homepage.path");

            ReadHighlightColors();

            c = new FrameWebView2(TxtHomepageURL.Text);
            editorInput = new FrameWebView2("", true);

            //years for wayback
            for (int year = 1994; year < DateTime.Now.Year; ++year)
            {
                CboWaybackYears.Items.Add(year.ToString());
            }

            //add any important event handlers here
            c.Dock = DockStyle.Fill;
            TabDocument.Controls.Add(c);
            editorInput.Dock = DockStyle.Fill;
            PanEditor.Controls.Add(editorInput);
            //editorInput.Browser.LoadHtml("<html><body contentEditable='true'></body></html>", "http://yourinputdoc");

            c.Browser.NavigationCompleted += Browser_NavigationCompleted;   //LoadingStateChanged += Browser_LoadingStateChanged;
            //c.Browser.LifeSpanHandler = new PopupHandler(lastURL);

            //zedgraph visuals

            //clickbait
            GraphPane clickbaitPane = new GraphPane();
            clickbaitPane.Title.Text = "The Experimental Clickbait Detector thinks the input text is...";
            clickbaitPane.YAxis.Title = new AxisLabel("Amount", "Consolas", 12, Color.Black, true, false, false);
            clickbaitPane.XAxis.Title = new AxisLabel("Clickbait Detection Scores", "Consolas", 12, Color.Black, true, false, false);
            clickbaitPane.XAxis.Type = AxisType.Text;
            string[] xAxisLabels = new string[] { "Not CB", "Slightly CB", "Moderate CB", "Heavy CB" };
            clickbaitPane.XAxis.Scale.TextLabels = xAxisLabels;
            clickbaitPane.AxisChange();
            zedGraphClickbait.GraphPane = clickbaitPane;

            //forces the clickbait graph to fit the panel properly on startup
            zedGraphClickbait.Dock = DockStyle.Fill;

            //satire
            GraphPane satirePane = new GraphPane();
            satirePane.Title.Text = "The Experimental Satire Detector thinks the input text is...";
            satirePane.YAxis.Title = new AxisLabel("Amount", "Consolas", 12, Color.Black, true, false, false);
            satirePane.XAxis.Title = new AxisLabel("Satire Detection Scores", "Consolas", 12, Color.Black, true, false, false);
            satirePane.XAxis.Type = AxisType.Text;
            string[] satireTitles = { "Not Satire", "Satire" };
            satirePane.XAxis.Scale.TextLabels = satireTitles;
            satirePane.XAxis.Type = AxisType.Text;
            satirePane.XAxis.Scale.FontSpec.Size = 14.0f;
            satirePane.AxisChange();
            zedGraphSatire.GraphPane = satirePane;

            //falsifications
            GraphPane falsificationPane = new GraphPane();
            falsificationPane.Title.Text = "The Experimental Falsified News Detector thinks the input text is...";
            falsificationPane.YAxis.Title = new AxisLabel("Amount", "Consolas", 12, Color.Black, true, false, false);
            falsificationPane.XAxis.Title = new AxisLabel("Falsification Detection Scores", "Consolas", 12, Color.Black, true, false, false);
            falsificationPane.XAxis.Type = AxisType.Text;
            string[] falsificationTitles = { "Legitimate", "Falsified" };
            falsificationPane.XAxis.Scale.TextLabels = falsificationTitles;
            falsificationPane.XAxis.Type = AxisType.Text;
            falsificationPane.XAxis.Scale.FontSpec.Size = 14.0f;
            falsificationPane.AxisChange();
            zedGraphFalsification.GraphPane = falsificationPane;

            string scriptPath = @"C:\Users\DHANASHREE\source\repos\brogly\WebResearch\WebResearchManager\";
            using (Py.GIL())
            {
                //clickbait
                pyDill = Py.Import("dill");
                pyClickbaitMl = Py.Import("clickbaitml");
                pyNvsClickbait = Py.Import("nvsclickbait");

                pyClickbaitDetectorDUMP = pyNvsClickbait.opendill();
                pyClickbaitDetector = pyDill.load(pyClickbaitDetectorDUMP);

                //public health


                //health relatedness


                //humor


                //hyperpartisian
                dynamic sys = Py.Import("sys");
                sys.path.append(scriptPath);

                pyHyperpartisanMl = Py.Import("PyMLHyperpartisan");
                
                pyHyperpartisanDetectorDUMP = pyHyperpartisanMl.opendill();
                pyHyperpartisanDetector = pyHyperpartisanDetectorDUMP;

                //political bias


                //fake news

            }

        }

        private async void Browser_NavigationCompleted(object? sender, CoreWebView2NavigationCompletedEventArgs e)
        {
            ncb = new List<string>();
            scb = new List<string>();
            mcb = new List<string>();
            hcb = new List<string>();

            var htmlJSON = await c.Browser.CoreWebView2.ExecuteScriptAsync("document.body.outerHTML");

            if (htmlJSON == "null")
            {
                return;
            }

            var html = JsonConvert.DeserializeObject(htmlJSON).ToString();

            //lastURL = e.Browser.MainFrame.Url;
            Invoke(new FocusBrowserTabCallback(FocusBrowserTab), new object[] { });

            //update UI
            //if (LblDetectorFalsified.InvokeRequired && LblDetectorLegit.InvokeRequired
            //&& numObservedFalsifiedScore.InvokeRequired && numObservedLegitScore.InvokeRequired && cboFalsificationDetectorAcc.InvokeRequired)
            {
                Invoke(new UpdateFalsificationDetectorMsgCallback(UpdateFalsificationDetectorMsg), new object[] { "Processing...", "Processing..." });
            }
            //if (LblDetectorSatire.InvokeRequired && LblDetectorNotSatire.InvokeRequired
            //&& numObservedNotSatire.InvokeRequired && numObservedSatire.InvokeRequired && cboSatireDetectorAcc.InvokeRequired)
            {
                Invoke(new UpdateSatireDetectorMsgCallback(UpdateSatireDetectorMsg), new object[] { "Processing...", "Processing..." });
            }

            //clickbait
            //if (clickbaitPanel.InvokeRequired)
            {
                Invoke(new ClearTextResultsCallback(ClearTextResults), new object[] { });
            }
            //if (txtBottomLine.InvokeRequired)
            {
                Invoke(new ClearBottomLineCallback(ClearBottomLine), new object[] { });
            }

            //satire
            //if (rtbSatireResults.InvokeRequired)
            {
                Invoke(new ClearSatireTextResultsCallback(ClearSatireTextResults), new object[] { });
            }
            //if (txtSatireFeatureScores.InvokeRequired)
            {
                Invoke(new TxtSatireResultsCallback(TxtSatireResults), new object[] { "" });
            }

            //falsifications
            //if (rtbFalsificationResults.InvokeRequired)
            {
                Invoke(new ClearFalsificationTextResultsCallback(ClearFalsificationTextResults), new object[] { });
            }
            //if (txtFalsificationFeatureScores.InvokeRequired)
            {
                Invoke(new TxtFalsificationResultsCallback(TxtFalsificationResults), new object[] { "" });
            }

            //have to manually clear the clickbait graph when we load a new page, not sure why
            //if (zedGraphSatire.InvokeRequired)
            {
                Invoke(new ClearSatireGraphCallback(ClearSatireGraph), new object[] { });
            }

            //if (zedGraphFalsification.InvokeRequired)
            {
                Invoke(new ClearFalsificationGraphCallback(ClearFalsificationGraph), new object[] { });
            }

            List<string> clickbaitResults = new List<string>();
            List<string> satireTextResults = new List<string>();
            List<string> falsificationTextResults = new List<string>();

            doc = new HtmlAgilityPack.HtmlDocument();
            doc.LoadHtml(html);

            int wordCountValue = GetSliderVal();

            //NOTE: crash here if there are no links
            //clickbait
            HtmlNodeCollection potentialClickbaitLinks = doc.DocumentNode.SelectNodes(clickbaitTagWnd.GetXPathTags());
            if (potentialClickbaitLinks != null)
            {
                foreach (HtmlNode link in potentialClickbaitLinks)
                {
                    if (clickbaitTagWnd.TxtHTMLID.Text != "")
                    {
                        if (link.Id != clickbaitTagWnd.TxtHTMLID.Text)
                        {
                            continue;
                        }
                    }

                    if (clickbaitTagWnd.TxtHTMLClass.Text != "")
                    {
                        if (link.HasClass(clickbaitTagWnd.TxtHTMLID.Text) == false)
                        {
                            continue;
                        }
                    }

                    // Get the value of the HREF attribute
                    // Ignore items < 3. These *may* be clickbait but they tend not to be.
                    string hrefValue = link.InnerText.Trim();
                    hrefValue = ReplaceSpace(hrefValue);

                    if (hrefValue != "" && hrefValue.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Length >= wordCountValue)
                    {
                        //this check for letters/digits should stop whitespace from being added
                        //double check this later - not 100% sure why this works
                        foreach (char ch in hrefValue)
                        {
                            if (Char.IsLetterOrDigit(ch))
                            {
                                //does not process duplicates
                                if (clickbaitResults.Contains(hrefValue) == false && (TxtCBFilter.Text == "" || hrefValue.Contains(TxtCBFilter.Text)))
                                    clickbaitResults.Add(hrefValue);
                                break;
                            }
                        }
                    }
                }
            }

            //satire
            int satireParagraphWordRequirement = 10;
            var satireParagraphNodes = doc.DocumentNode.SelectNodes(satireTagWnd.GetXPathTags());
            if (satireParagraphNodes != null)
            {
                foreach (HtmlNode paragraph in satireParagraphNodes)
                {
                    if (satireTagWnd.TxtHTMLID.Text != "")
                    {
                        if (paragraph.Id != satireTagWnd.TxtHTMLID.Text)
                        {
                            continue;
                        }
                    }

                    if (satireTagWnd.TxtHTMLClass.Text != "")
                    {
                        if (paragraph.HasClass(satireTagWnd.TxtHTMLID.Text) == false)
                        {
                            continue;
                        }
                    }

                    // Get the value of the HREF attribute
                    // Ignore items < 3. These *may* be clickbait but they tend not to be.
                    string paragraphValue = paragraph.InnerText.Trim();
                    paragraphValue = ReplaceSpace(paragraphValue);

                    if (SATIRE_TAG_TEXT_MIN_CHAR_LENGTH > 0)
                    {
                        if (paragraphValue.Length < SATIRE_TAG_TEXT_MIN_CHAR_LENGTH)
                        {
                            continue;
                        }
                    }

                    if (paragraphValue != "" && paragraphValue.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Length > satireParagraphWordRequirement)
                    {
                        satireTextResults.Add(paragraphValue);
                    }
                }
            }

            //falsifications
            int falsificationParagraphWordRequirement = 10;
            var falsificationParagraphNodes = doc.DocumentNode.SelectNodes(falsificationTagWnd.GetXPathTags());
            if (falsificationParagraphNodes != null)
            {
                foreach (HtmlNode paragraph in falsificationParagraphNodes)
                {
                    if (falsificationTagWnd.TxtHTMLID.Text != "")
                    {
                        if (paragraph.Id != falsificationTagWnd.TxtHTMLID.Text)
                        {
                            continue;
                        }
                    }

                    if (falsificationTagWnd.TxtHTMLClass.Text != "")
                    {
                        if (paragraph.HasClass(falsificationTagWnd.TxtHTMLID.Text) == false)
                        {
                            continue;
                        }
                    }

                    // Get the value of the HREF attribute
                    // Ignore items < 3. These *may* be clickbait but they tend not to be.
                    string paragraphValue = paragraph.InnerText.Trim();
                    paragraphValue = ReplaceSpace(paragraphValue);

                    if (FALSIFICATION_TAG_TEXT_MIN_CHAR_LENGTH > 0)
                    {
                        if (paragraphValue.Length < FALSIFICATION_TAG_TEXT_MIN_CHAR_LENGTH)
                        {
                            continue;
                        }
                    }

                    if (paragraphValue != "" && paragraphValue.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Length > falsificationParagraphWordRequirement)
                    {
                        falsificationTextResults.Add(paragraphValue);
                    }
                }
            }

            //if (txtURL.InvokeRequired)
            {
                Invoke(new UpdateURLCallback(UpdateURL), new object[] { c.Browser.Source.ToString() });
            }

            DetectClickbait(clickbaitResults);
            //DetectSatire(satireTextResults);
            //DetectFalsification(falsificationTextResults);

            if (CrawlMode == true)
            {
                TxtPageSaveURL.Text = txtURL.Text;

                bool clickbait = chkClickbait.Checked;
                bool satire = chkSatire.Checked;
                bool falsifications = chkFalsification.Checked;

                SaveResultsToDB(clickbait, satire, falsifications, true);

                ++crawlIndex;
                if (crawlIndex == crawlURLs.Count)
                {
                    crawlIndex = 0;
                    CrawlMode = false;

                    CrawlMode = false;
                    BtnSetCrawlMode.Text = "Crawl Mode is OFF";
                    BtnSetCrawlMode.BackColor = Color.DarkRed;
                    crawlURLs = new List<string>();
                    RtbCrawlLinkList.ReadOnly = false;
                    TxtCrawlOutput.Text += "[" + DateTime.Now.ToString() + "] " + "Crawls complete.";
                }
                else
                {
                    TxtCrawlOutput.Text += "[" + DateTime.Now.ToString() + "] " + "Crawling " + crawlURLs[crawlIndex] + " ... " + Environment.NewLine;
                    c.Navigate(crawlURLs[crawlIndex]);
                }
            }
        }

        private void ClickbaitView_MouseDoubleClick(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                frmStats.ShowClickbaitDetails(clickbaitPanel.GetEntry(Convert.ToInt32(clickbaitPanel.LstViewClickbait.SelectedItems[0].SubItems[0].Text)));
                frmStats.ShowOtherClickbaitDetails(clickbaitPanel.GetEntries());
                frmStats.Show();
                frmStats.BringToFront();
            }
        }

        private void ClickbaitView_MouseClick(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                //note: this won't work all the time because the Text won't nessecarily match what is on the page due to encoding
                //c.Browser.GetBrowser().Find(0, clickbaitPanel.LstViewClickbait.SelectedItems[0].SubItems[4].Text, true, false, false);
            }
            else if (e.Button == MouseButtons.Right)
            {
                List<string> cbEntry = clickbaitPanel.GetEntry(Convert.ToInt32(clickbaitPanel.LstViewClickbait.SelectedItems[0].SubItems[0].Text));
                string id = clickbaitPanel.LstViewClickbait.SelectedItems[0].SubItems[0].Text;
                string cbScore = cbEntry[0];
                string ncbScore = cbEntry[1];
                string caption = cbEntry[cbEntry.Count - 1];

                FrmCBEntryUserScore cbUserScore = new FrmCBEntryUserScore(id, caption, cbScore, ncbScore,
                    clickbaitPanel.ClickbaitUserScoreResults[Convert.ToInt32(id)][0], clickbaitPanel.ClickbaitUserScoreResults[Convert.ToInt32(id)][1],
                    clickbaitPanel.ClickbaitUserScoreResults[Convert.ToInt32(id)][2]);
                List<string> newUserScores = cbUserScore.GetUserScores();
                clickbaitPanel.ClickbaitUserScoreResults[Convert.ToInt32(id)] = new List<string>(newUserScores);
            }
        }

        const int CLICKBAIT_SCORE = 0;
        const int NOT_CLICKBAIT_SCORE = 1;
        const int HEADLINE_STRING = 2;
        const int NOT_CLICKBAIT = 0;
        const int SLIGHT_CLICKBAIT = 1;
        const int MODERATE_CLICKBAIT = 2;
        const int HEAVY_CLICKBAIT = 3;

        public static int SATIRE_TAG_TEXT_MIN_CHAR_LENGTH = 0;
        public static int FALSIFICATION_TAG_TEXT_MIN_CHAR_LENGTH = 0;

        public static double CLICKBAIT_SLIGHT_THRESHOLD = 0.15;
        public static double CLICKBAIT_MODERATE_THRESHOLD = 0.35;
        public static double CLICKBAIT_HEAVY_THRESHOLD = 0.7;

        public static string CLICKBAIT_LOW_COLOR;
        public static string CLICKBAIT_SLIGHT_COLOR;
        public static string CLICKBAIT_MODERATE_COLOR;
        public static string CLICKBAIT_HEAVY_COLOR;

        //not cb, slightly cb, moderate cb, heavy CB, the lists are used for background coloring to show clickbait severity
        private double[] yVals = { 0, 0, 0, 0 };
        private List<string> ncb = new List<string>();
        private List<string> scb = new List<string>();
        private List<string> mcb = new List<string>();
        private List<string> hcb = new List<string>();

        private void ClickbaitOutputLine(string outStr)
        {
            //need this otherwise error output wont show
            if (outStr == null)
                return;

            Debug.WriteLine(outStr);

            string clickbaitScoreOutput = outStr;

            //if (clickbaitPanel.InvokeRequired)
            {
                Invoke(new AddLineCallBack(AddClickbaitLine), new object[] { clickbaitScoreOutput });
            }

            string headline = clickbaitScoreOutput;
            int startOfHeadlineText;
            int commaCount = 0;
            for (startOfHeadlineText = 0; startOfHeadlineText < headline.Length; ++startOfHeadlineText)
            {
                if (commaCount == 40)
                {
                    break;
                }
                if (headline[startOfHeadlineText] == ',')
                {
                    commaCount++;
                }
            }

            string[] lnItems = clickbaitScoreOutput.Split(',');
            double cbScore = Convert.ToDouble(lnItems[CLICKBAIT_SCORE]);
            if (cbScore >= CLICKBAIT_HEAVY_THRESHOLD)
            {
                yVals[HEAVY_CLICKBAIT]++;
                hcb.Add(headline.Substring(startOfHeadlineText).Replace("\"", "\\\""));
            }
            else if (cbScore >= CLICKBAIT_MODERATE_THRESHOLD)
            {
                yVals[MODERATE_CLICKBAIT]++;
                mcb.Add(headline.Substring(startOfHeadlineText).Replace("\"", "\\\""));
            }
            else if (cbScore >= CLICKBAIT_SLIGHT_THRESHOLD)
            {
                yVals[SLIGHT_CLICKBAIT]++;
                scb.Add(headline.Substring(startOfHeadlineText).Replace("\"", "\\\""));
            }
            else if (cbScore > 0)
            {
                yVals[NOT_CLICKBAIT]++;
                ncb.Add(headline.Substring(startOfHeadlineText).Replace("\"", "\\\""));
            }

            double amtOfHeadlines = yVals.Sum();
            double amtOfClickbait = yVals[MODERATE_CLICKBAIT] + yVals[HEAVY_CLICKBAIT];
            double clickbaitPercentage = amtOfClickbait / amtOfHeadlines;

            string overallPage = "";

            if (clickbaitPercentage >= CLICKBAIT_HEAVY_THRESHOLD)
            {
                overallPage = "Page is heavily clickbaiting.";
            }
            else if (clickbaitPercentage >= CLICKBAIT_MODERATE_THRESHOLD)
            {
                overallPage = "Page is moderately clickbaiting.";
            }
            else if (clickbaitPercentage >= CLICKBAIT_SLIGHT_THRESHOLD)
            {
                overallPage = "Page is slightly clickbaiting.";
            }
            else
            {
                overallPage = "Page clickbaiting level is low.";
            }

            //if (zedGraphClickbait.InvokeRequired)
            {
                Invoke(new UpdateClickbaitGraphCallback(UpdateClickbaitGraph), new object[] { yVals });
            }
            //if (txtBottomLine.InvokeRequired)
            {
                Invoke(new BottomLineCallback(BottomLine), new object[] { overallPage + " (" + Math.Round(clickbaitPercentage * 100) + " % of links) Links: " + yVals.Sum() });
            }

            Invoke(new ClickbaitHighlightCallBack(ClickbaitHighlight), new object[] { });
        }

        private void ClickbaitErrorLine(object s, DataReceivedEventArgs ea)
        {
            this.Invoke((MethodInvoker)delegate
            {
                frmErrors.WriteErrorLn("Clickbait Error Output: " + ea.Data);
                frmErrors.Show();
            });
            Debug.WriteLine("Clickbait Error Output: " + ea.Data);
        }

        delegate void UpdateSatireDetectorScoresCallback(double l, double f);
        private void UpdateSatireDetectorScores(double legit, double fals)
        {
            LblDetectorNotSatire.Text = (Math.Round(legit, 2) * 100).ToString();
            LblDetectorSatire.Text = (Math.Round(fals, 2) * 100).ToString();
        }

        delegate void UpdateSatireDetectorMsgCallback(string l, string f);
        private void UpdateSatireDetectorMsg(string legit, string fals)
        {
            LblDetectorNotSatire.Text = legit;
            LblDetectorSatire.Text = fals;
            numObservedNotSatire.Value = 0;
            numObservedSatire.Value = 0;
            cboSatireDetectorAcc.SelectedIndex = 0;
        }

        public static string SATIRE_TEXT_COLOR;
        public static string NOT_SATIRE_TEXT_COLOR;

        public static string FALSIFICATION_TEXT_COLOR;
        public static string NOT_FALSIFICATION_TEXT_COLOR;

        private static string CLEAR_TEXT_COLOR = "255,255,255";

        private bool SATIRE_MOSTLY_LEGIT = true;
        private bool FALSIFICATION_MOSTLY_LEGIT = true;

        private void SatireOutputLine(string outStr)
        {
            //need this otherwise error output wont show
            if (outStr == null)
                return;

            Debug.WriteLine(outStr);

            string satireScoreOutput = outStr;

            string[] splitSatireScores = satireScoreOutput.Split(',');

            //output format is PROBABILITY_SCORE, SATIRE_FLAG (0,1), ABSURDITY_FLAG, (0,1), HUM_FLAG, (0,1)
            double[] yVals = { Convert.ToDouble(splitSatireScores[0]), Convert.ToDouble(splitSatireScores[1]) };

            if (yVals[0] < 0.5)
                SATIRE_MOSTLY_LEGIT = false;
            else
                SATIRE_MOSTLY_LEGIT = true;

            //absurd and humor flags have been commented out since they are disabled in the detector right now... need to fix them there.
            //int absurdFlag = Convert.ToInt32(splitSatireScores[2]);
            //int humorFlag = Convert.ToInt32(splitSatireScores[3]);

            //if (zedGraphSatire.InvokeRequired)
            {
                Invoke(new UpdateSatireGraphCallback(UpdateSatireGraph), new object[] { yVals });
            }
            //if (LblDetectorSatire.InvokeRequired && LblDetectorNotSatire.InvokeRequired)
            {
                Invoke(new UpdateSatireDetectorScoresCallback(UpdateSatireDetectorScores), new object[] { yVals[0], yVals[1] });
            }
            //if (txtSatireFeatureScores.InvokeRequired)
            {
                Invoke(new TxtSatireResultsCallback(TxtSatireResults), new object[] { satireScoreOutput });
            }
            Invoke(new SatireHighlightCallBack(SatireHighlight), new object[] { });
        }

        private void SatireErrorLine(object s, DataReceivedEventArgs ea)
        {
            this.Invoke((MethodInvoker)delegate
            {
                frmErrors.WriteErrorLn("Satire Error Output: " + ea.Data);
                frmErrors.Show();
            });
            Debug.WriteLine("Satire Error Output: " + ea.Data);
        }

        delegate void UpdateFalsificationDetectorScoresCallback(double l, double f);
        private void UpdateFalsificationDetectorScores(double legit, double fals)
        {
            LblDetectorLegit.Text = (Math.Round(legit, 2) * 100).ToString();
            LblDetectorFalsified.Text = (Math.Round(fals, 2) * 100).ToString();
        }

        delegate void UpdateFalsificationDetectorMsgCallback(string l, string f);
        private void UpdateFalsificationDetectorMsg(string legit, string fals)
        {
            LblDetectorLegit.Text = legit;
            LblDetectorFalsified.Text = fals;
            numObservedFalsifiedScore.Value = 0;
            numObservedLegitScore.Value = 0;
            cboFalsificationDetectorAcc.SelectedIndex = 0;
        }

        private void FalsificationOutputLine(string outStr)
        {
            //need this otherwise error output wont show
            if (outStr == null)
                return;

            Debug.WriteLine(outStr);

            string falsificationScoreOutput = outStr;

            string[] splitFalsificationScores = falsificationScoreOutput.Split(',');

            double[] yVals = { Convert.ToDouble(splitFalsificationScores[0]), Convert.ToDouble(splitFalsificationScores[1]) };

            if (yVals[0] < 0.5)
                FALSIFICATION_MOSTLY_LEGIT = false;
            else
                FALSIFICATION_MOSTLY_LEGIT = true;

            //if (zedGraphFalsification.InvokeRequired)
            {
                Invoke(new UpdateFalsificationGraphCallback(UpdateFalsificationGraph), new object[] { yVals });
            }
            //if (LblDetectorFalsified.InvokeRequired && LblDetectorLegit.InvokeRequired)
            {
                Invoke(new UpdateFalsificationDetectorScoresCallback(UpdateFalsificationDetectorScores), new object[] { yVals[0], yVals[1] });
            }
            //if (txtFalsificationFeatureScores.InvokeRequired)
            {
                Invoke(new TxtFalsificationResultsCallback(TxtFalsificationResults), new object[] { falsificationScoreOutput });
            }

            Invoke(new FalsificationHighlightCallBack(FalsificationHighlight), new object[] { });
        }

        private void FalsificationErrorLine(object s, DataReceivedEventArgs ea)
        {
            this.Invoke((MethodInvoker)delegate
            {
                frmErrors.WriteErrorLn("Falsification Error Output: " + ea.Data);
                frmErrors.Show();
            });
            Debug.WriteLine("Falsification Error Output: " + ea.Data);
        }

        private void StartClickbaitDetector()
        {
            if (!clickbaitDetector.Start())
            {
                Debug.WriteLine("Failed to start Clickbait Detector");
            }
            else
            {
                Debug.WriteLine("Clickbait Detector Started");
            }
        }

        private void StartSatireDetector()
        {
            if (!satireDetector.Start())
            {
                Debug.WriteLine("Failed to start Satire Detector");
            }
            else
            {
                Debug.WriteLine("Satire Detector Started");
            }
        }

        private void StartFalsificationDetector()
        {
            if (!falsificationDetector.Start())
            {
                Debug.WriteLine("Failed to start Falsification Detector");
            }
            else
            {
                Debug.WriteLine("Falsification Detector Started");
            }
        }

        //some pages tend to load over and over again
        //keeping track of the last URL prevents that
        private string lastURL = "";

        private string ReplaceSpace(string hrefValue)
        {
            string newHrefValue = "";
            bool spaceFlag = false;
            for (int i = 0; i < hrefValue.Length; ++i)
            {
                if (hrefValue[i] == ' ' && spaceFlag == false)
                {
                    newHrefValue += hrefValue[i];
                    spaceFlag = true;
                }
                else if (hrefValue[i] == '\r' || hrefValue[i] == '\n' || hrefValue[i] == '\t')
                    continue;
                else
                {
                    newHrefValue += hrefValue[i];
                    spaceFlag = false;
                }
            }
            return newHrefValue;
        }

        private void DetectClickbait(List<string> links)
        {
            //StartClickbaitDetector();

            yVals = new double[] { 0, 0, 0, 0 };

            /* Previously we used synchronous I/O here and for some reason
             * there was always a random stop here with no exception or error message if there were too many links
             * Asynchronous output seems to work now, so OutputLine and ErrorLine handle things now */

            List<string> linkCaptions = new List<string>();
            for (int i = 0; i < links.Count; ++i)
            {
                linkCaptions.Add(links[i]);
            }
            foreach (string s in linkCaptions)
            {
                if (!String.IsNullOrEmpty(s))
                {
                    using (Py.GIL())
                    {
                        Debug.WriteLine(s);

                        //clickbait
                        //run pythonnet for detectors
                        string result = (string)pyClickbaitDetector.predict(s, false);

                        //public health


                        //health relatedness


                        //humor


                        //hyperpartisian
                        dynamic hypresult = pyHyperpartisanDetector.predict(s);
                        Console.WriteLine(hypresult);
                        //political bias


                        //fake news


                        string[] dRes = s.Split(",");

                        try
                        {
                            ClickbaitOutputLine(result);
                        }
                        catch (Exception ex)
                        {
                            MessageBox.Show(ex.Message);
                        }
                    }
                }
            }
        }

        private void DetectSatire(List<string> texts)
        {
            string text = "";
            for (int i = 0; i < texts.Count; ++i)
            {
                string trimmedText = ReplaceSpace(texts[i]);

                //if (rtbSatireResults.InvokeRequired)
                {
                    Invoke(new AddSatireResultLineCallBack(AddSatireResultLine), new object[] { trimmedText });
                }

                text += trimmedText;
            }

            if (!String.IsNullOrEmpty(text))
            {
                Debug.WriteLine(text);
                satireDetector.StandardInput.WriteLine(text);
                try
                {
                    SatireOutputLine(satireDetector.StandardOutput.ReadLine());
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }
            }
            else
            {
                //update labels to show no text was found
                Invoke(new UpdateSatireDetectorLabelsCallback(UpdateSatireDetectorLabels), new object[] { });
            }
        }

        private void DetectFalsification(List<string> texts)
        {
            string text = "";
            for (int i = 0; i < texts.Count; ++i)
            {
                string trimmedText = ReplaceSpace(texts[i]);

                //if (rtbFalsificationResults.InvokeRequired)
                {
                    Invoke(new AddFalsificationResultLineCallBack(AddFalsificationResultLine), new object[] { trimmedText });
                }

                text += trimmedText;
            }

            if (!String.IsNullOrEmpty(text))
            {
                Debug.WriteLine(text);
                falsificationDetector.StandardInput.WriteLine(text);
                try
                {
                    FalsificationOutputLine(falsificationDetector.StandardOutput.ReadLine());
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }
            }
            else
            {
                //update labels to show no text was found
                Invoke(new UpdateFalsificationDetectorLabelsCallback(UpdateFalsificationDetectorLabels), new object[] { });
            }
        }

        //NOTE: the code for updating the UI here is mostly a copy + paste and could be refactored or a pattern could be used here

        #region "Clickbait Panel"

        delegate void UpdateClickbaitGraphCallback(double[] yVals);

        private void UpdateClickbaitGraph(double[] yVals)
        {
            zedGraphClickbait.GraphPane.CurveList.Clear();
            zedGraphClickbait.GraphPane.GraphObjList.Clear();

            //the following code is somewhat confusing - the pointpairlist makes things easier
            double[] xVals = new double[] { 1.0, 2.0, 3.0, 4.0 };

            string[] splitColor;

            splitColor = CLICKBAIT_LOW_COLOR.Split(',');
            Color notCB = Color.FromArgb(Convert.ToInt32(splitColor[0]), Convert.ToInt32(splitColor[1]), Convert.ToInt32(splitColor[2]));
            splitColor = CLICKBAIT_SLIGHT_COLOR.Split(',');
            Color slightCB = Color.FromArgb(Convert.ToInt32(splitColor[0]), Convert.ToInt32(splitColor[1]), Convert.ToInt32(splitColor[2]));
            splitColor = CLICKBAIT_MODERATE_COLOR.Split(',');
            Color moderateCB = Color.FromArgb(Convert.ToInt32(splitColor[0]), Convert.ToInt32(splitColor[1]), Convert.ToInt32(splitColor[2]));
            splitColor = CLICKBAIT_HEAVY_COLOR.Split(',');
            Color heavyCB = Color.FromArgb(Convert.ToInt32(splitColor[0]), Convert.ToInt32(splitColor[1]), Convert.ToInt32(splitColor[2])); ;

            Color[] colors = { notCB, slightCB, moderateCB, heavyCB };
            PointPairList coords = new PointPairList(xVals, yVals);

            //then we add all of the point pairs to create 4 purple bars
            BarItem bars = zedGraphClickbait.GraphPane.AddBar("LinkCount", coords, Color.Purple);

            //now we separate them into 4 colours for the 4 categories
            bars.Bar.Fill = new Fill(colors);
            bars.Bar.Fill.Type = FillType.GradientByX;

            //this will color the bars as follows
            //the first will be Green (x-val 1), second yellow (x-val 2), third organge (x-val 3), fourth red (x-val 4)
            //the RangeMin and Max are confusing but they work OK
            bars.Bar.Fill.RangeMin = 1;
            bars.Bar.Fill.RangeMax = 4;

            zedGraphClickbait.AxisChange();

            BarItem.CreateBarLabels(zedGraphClickbait.GraphPane, true, "", "", 16, Color.Black, true, false, false);
            foreach (TextObj t in zedGraphClickbait.GraphPane.GraphObjList)
            {
                t.FontSpec.Angle = 0;
            }

            zedGraphClickbait.Refresh();
        }

        private double[] dblArr(double val)
        {
            double[] arr = new double[1];
            arr[0] = val;
            return arr;
        }

        delegate int GetSliderValueCallback();
        private int GetSliderVal()
        {
            if (NumMinWordCount.InvokeRequired)
            {
                GetSliderValueCallback cb = new GetSliderValueCallback(GetSliderVal);
                return (int)NumMinWordCount.Invoke(cb);
            }
            else
            {
                return (int)NumMinWordCount.Value;
            }
        }

        delegate void UpdateURLCallback(string url);

        private void UpdateURL(string url)
        {
            txtURL.Text = url;
        }

        delegate void AddLineCallBack(string line);

        private void AddClickbaitLine(string ln)
        {
            clickbaitPanel.AddClickbaitToListViewAndDict(ln);
        }

        delegate void ClearTextResultsCallback();

        private void ClearTextResults()
        {
            clickbaitPanel.Reset();
        }

        delegate void BottomLineCallback(string line);

        private void BottomLine(string ln)
        {
            txtBottomLine.Text = ln;
        }

        delegate void ClearBottomLineCallback();

        private void ClearBottomLine()
        {
            txtBottomLine.Text = "Waiting for clickbait input...";
        }

        delegate void UpdateSatireDetectorLabelsCallback();

        private void UpdateSatireDetectorLabels()
        {
            LblDetectorSatire.Text = "No text.";
            LblDetectorNotSatire.Text = "No text.";
        }

        delegate void UpdateFalsificationDetectorLabelsCallback();

        private void UpdateFalsificationDetectorLabels()
        {
            LblDetectorFalsified.Text = "No text.";
            LblDetectorLegit.Text = "No text.";
        }

        delegate void FocusBrowserTabCallback();

        private void FocusBrowserTab()
        {
            TabsBrowser.SelectTab(0);
        }

        #endregion

        #region "Satire Panel"

        delegate void AddSatireResultLineCallBack(string line);

        private void AddSatireResultLine(string ln)
        {
            rtbSatireResults.Text += ln;
        }

        delegate void ClearSatireTextResultsCallback();

        private void ClearSatireTextResults()
        {
            rtbSatireResults.Clear();
        }

        delegate void TxtSatireResultsCallback(string line);

        private void TxtSatireResults(string ln)
        {
            txtSatireFeatureScores.Text = ln;
        }

        delegate void UpdateSatireGraphCallback(double[] yVals);

        private void UpdateSatireGraph(double[] yVals)
        {
            zedGraphSatire.GraphPane.CurveList.Clear();
            zedGraphSatire.GraphPane.GraphObjList.Clear();

            //the following code is somewhat confusing - the pointpairlist makes things easier
            double[] xVals = new double[] { 1.0, 2.0 };

            string[] splitColor;
            splitColor = SATIRE_TEXT_COLOR.Split(',');
            Color falseColor = Color.FromArgb(Convert.ToInt32(splitColor[0]), Convert.ToInt32(splitColor[1]), Convert.ToInt32(splitColor[2]));
            splitColor = NOT_SATIRE_TEXT_COLOR.Split(',');
            Color legitColor = Color.FromArgb(Convert.ToInt32(splitColor[0]), Convert.ToInt32(splitColor[1]), Convert.ToInt32(splitColor[2]));

            Color[] colors = { legitColor, falseColor };

            PointPairList coords = new PointPairList(xVals, yVals);

            //then we add all of the point pairs to create 4 purple bars
            BarItem bars = zedGraphSatire.GraphPane.AddBar("SatireScore", coords, Color.Purple);

            //now we separate them into 4 colours for the 4 categories
            bars.Bar.Fill = new Fill(colors);
            bars.Bar.Fill.Type = FillType.GradientByX;

            //this will color the bars as follows
            //the first will be Green (x-val 1), second yellow (x-val 2), third organge (x-val 3), fourth red (x-val 4)
            //the RangeMin and Max are confusing but they work OK
            bars.Bar.Fill.RangeMin = 1;
            bars.Bar.Fill.RangeMax = 2;

            zedGraphSatire.AxisChange();

            BarItem.CreateBarLabels(zedGraphSatire.GraphPane, true, "", "", 16, Color.Black, true, false, false);
            foreach (TextObj t in zedGraphSatire.GraphPane.GraphObjList)
            {
                t.FontSpec.Angle = 0;
            }

            zedGraphSatire.Refresh();
        }

        delegate void ClearSatireGraphCallback();

        //unlike the clickbait graph, the satire graph does not clear when you get new info, so clear it manually here
        private void ClearSatireGraph()
        {
            zedGraphSatire.GraphPane.CurveList.Clear();
            zedGraphSatire.GraphPane.GraphObjList.Clear();
            zedGraphSatire.AxisChange();
            zedGraphSatire.Refresh();
        }

        #endregion

        #region "Falsification Panel"

        delegate void AddFalsificationResultLineCallBack(string line);

        private void AddFalsificationResultLine(string ln)
        {
            rtbFalsificationResults.Text += ln;
        }

        delegate void ClearFalsificationTextResultsCallback();

        private void ClearFalsificationTextResults()
        {
            rtbFalsificationResults.Clear();
        }

        delegate void TxtFalsificationResultsCallback(string line);

        private void TxtFalsificationResults(string ln)
        {
            txtFalsificationFeatureScores.Text = ln;
        }

        delegate void UpdateFalsificationGraphCallback(double[] yVals);

        private void UpdateFalsificationGraph(double[] yVals)
        {
            zedGraphFalsification.GraphPane.CurveList.Clear();
            zedGraphFalsification.GraphPane.GraphObjList.Clear();

            //the following code is somewhat confusing - the pointpairlist makes things easier
            double[] xVals = new double[] { 1.0, 2.0 };

            string[] splitColor;
            splitColor = FALSIFICATION_TEXT_COLOR.Split(',');
            Color falseColor = Color.FromArgb(Convert.ToInt32(splitColor[0]), Convert.ToInt32(splitColor[1]), Convert.ToInt32(splitColor[2]));
            splitColor = NOT_FALSIFICATION_TEXT_COLOR.Split(',');
            Color legitColor = Color.FromArgb(Convert.ToInt32(splitColor[0]), Convert.ToInt32(splitColor[1]), Convert.ToInt32(splitColor[2]));

            Color[] colors = { legitColor, falseColor };

            PointPairList coords = new PointPairList(xVals, yVals);

            //then we add all of the point pairs to create 4 purple bars
            BarItem bars = zedGraphFalsification.GraphPane.AddBar("FalsificationScore", coords, Color.Purple);

            //now we separate them into 4 colours for the 4 categories
            bars.Bar.Fill = new Fill(colors);
            bars.Bar.Fill.Type = FillType.GradientByX;

            //this will color the bars as follows
            //the first will be Green (x-val 1), second yellow (x-val 2), third organge (x-val 3), fourth red (x-val 4)
            //the RangeMin and Max are confusing but they work OK
            bars.Bar.Fill.RangeMin = 1;
            bars.Bar.Fill.RangeMax = 2;

            zedGraphFalsification.AxisChange();

            BarItem.CreateBarLabels(zedGraphFalsification.GraphPane, true, "", "", 16, Color.Black, true, false, false);
            foreach (TextObj t in zedGraphFalsification.GraphPane.GraphObjList)
            {
                t.FontSpec.Angle = 0;
            }

            zedGraphFalsification.Refresh();
        }

        delegate void ClearFalsificationGraphCallback();

        //unlike the clickbait graph, the satire graph does not clear when you get new info, so clear it manually here
        private void ClearFalsificationGraph()
        {
            zedGraphFalsification.GraphPane.CurveList.Clear();
            zedGraphFalsification.GraphPane.GraphObjList.Clear();
            zedGraphFalsification.AxisChange();
            zedGraphFalsification.Refresh();
        }

        #endregion

        private void BtnBack_Click(object sender, EventArgs e)
        {
            if (c.Browser.CoreWebView2.CanGoBack == true)
            {
                c.Browser.CoreWebView2.GoBack();
            }
        }

        private void BtnForward_Click(object sender, EventArgs e)
        {
            if (c.Browser.CoreWebView2.CanGoForward == true)
            {
                c.Browser.CoreWebView2.GoForward();
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            var aboutBox = new FrmAbout();
            aboutBox.ShowDialog();
        }

        private void txtURL_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Enter)
            {
                lastURL = "";
                c.Navigate(txtURL.Text);
                e.SuppressKeyPress = true;
                e.Handled = true;
            }
        }

        private void txtURL_DoubleClick(object sender, EventArgs e)
        {
            txtURL.SelectAll();
        }

        private void BtnDevTools_Click(object sender, EventArgs e)
        {
            c.Browser.CoreWebView2.OpenDevToolsWindow();
        }

        private void BtnRefresh_Click(object sender, EventArgs e)
        {
            lastURL = "";   //need to clear so refreshes work
            c.Browser.CoreWebView2.Reload();
        }

        private void BtnStatisticsWnd_Click(object sender, EventArgs e)
        {
            frmStats.Show();
            frmStats.BringToFront();
        }

        const int CLICKBAIT_CAPTION = 40;

        //ideally this should be updated so that multiple detectors can be accomodated instead of this all being hard-coded
        private void SaveResultsToDB(bool saveClickbait, bool saveSatire, bool saveFalsifications, bool saveNotes = false)
        {
            if (CboDatabases.Text == "")
            {
                MessageBox.Show("No dataset was selected!");
                return;
            }

            SQLiteConnection conn = new SQLiteConnection("Data Source=" + CboDatabases.Items[CboDatabases.SelectedIndex] + ";Version=3");
            conn.Open();

            int clickbaitSaved = 0;
            bool satireSaved = false;
            bool falsificationSaved = false;
            bool noteSaved = false;

            const int OBSERVED_NOT_CLICKBAIT = 0;
            const int OBSERVED_CLICKBAIT = 1;
            const int OBSERVED_CB_DETECTOR_ACC = 2;

            if (saveClickbait == true)
            {
                Dictionary<int, List<string>> clickbaitEntries = new Dictionary<int, List<string>>();
                Dictionary<int, List<string>> clickbaitUserScoreEntries = new Dictionary<int, List<string>>();

                clickbaitEntries = clickbaitPanel.GetEntries();
                clickbaitUserScoreEntries = clickbaitPanel.GetUserScoreEntries();

                for (int i = 0; i < clickbaitEntries.Count; ++i)
                {
                    string url;
                    if (TxtPageSaveURL.Text != null)
                    {
                        url = TxtPageSaveURL.Text.Replace("'", "''");
                    }
                    else
                    {
                        url = txtURL.Text.Replace("'", "''");
                    }

                    List<string> allCols = new List<string>(clickbaitEntries[i]);
                    List<string> allUserScoreCols = new List<string>(clickbaitUserScoreEntries[i]);

                    string clickbaitCaption = allCols[CLICKBAIT_CAPTION];
                    clickbaitCaption = clickbaitCaption.Replace("'", "''");

                    string insertClickbait = "insert into clickbait values(null, $url, $clickbaitCaption, $obsNotClickbait, $obsClickbait, $obsDetectorAcc, ";

                    //note: got some weird behaviour here probably do to reference types, this works though
                    for (int c = 0; c < 40; ++c)
                    {
                        if (c == 39)
                        {
                            insertClickbait += "$c" + Convert.ToString(c);
                        }
                        else
                        {
                            insertClickbait += "$c" + Convert.ToString(c) + ", ";
                        }
                    }

                    insertClickbait += ", $numMinWordCount, $xPathTags, $time, $date);";

                    SQLiteCommand cmd = new SQLiteCommand(insertClickbait, conn);
                    cmd.Parameters.AddWithValue("$url", url);
                    cmd.Parameters.AddWithValue("$clickbaitCaption", clickbaitCaption);
                    cmd.Parameters.AddWithValue("$obsNotClickbait", allUserScoreCols[OBSERVED_NOT_CLICKBAIT]);
                    cmd.Parameters.AddWithValue("$obsClickbait", allUserScoreCols[OBSERVED_CLICKBAIT]);
                    cmd.Parameters.AddWithValue("$obsDetectorAcc", allUserScoreCols[OBSERVED_CB_DETECTOR_ACC]);

                    //note: got some weird behaviour here probably do to reference types, this works though
                    for (int c = 0; c < 40; ++c)
                    {
                        cmd.Parameters.AddWithValue("$c" + c.ToString(), allCols[c]);
                    }

                    cmd.Parameters.AddWithValue("$numMinWordCount", NumMinWordCount.Value.ToString());
                    cmd.Parameters.AddWithValue("$xPathTags", clickbaitTagWnd.GetXPathTags());
                    cmd.Parameters.AddWithValue("$time", DateTime.Now.ToShortTimeString());
                    cmd.Parameters.AddWithValue("$date", DateTime.Now.ToShortDateString());

                    cmd.ExecuteNonQuery();
                    clickbaitSaved++;
                }
            }

            if (saveSatire == true)
            {
                if (txtSatireFeatureScores.Text != "")
                {
                    string url = txtURL.Text.Replace("'", "''");
                    string satireText = rtbSatireResults.Text;
                    satireText = satireText.Replace("'", "''");
                    string[] featureScores = txtSatireFeatureScores.Text.Split(',');

                    string insertSatire = String.Format("insert into satire values(null, $url, $satireText, $numObsNotSatire, $numObsSatire, $obsDetectorAcc,");

                    //string allColumns = String.Join(", ", featureScores);
                    for (int c = 0; c < featureScores.Length; ++c)
                    {
                        if (c == featureScores.Length - 1)
                        {
                            insertSatire += "$c" + Convert.ToString(c);
                        }
                        else
                        {
                            insertSatire += "$c" + Convert.ToString(c) + ", ";
                        }
                    }

                    //add cols here

                    insertSatire += ", $xPathTags, $time, $date);";

                    SQLiteCommand cmd = new SQLiteCommand(insertSatire, conn);

                    cmd.Parameters.AddWithValue("$url", url);
                    cmd.Parameters.AddWithValue("$satireText", satireText);
                    cmd.Parameters.AddWithValue("$numObsNotSatire", numObservedNotSatire.Value.ToString());
                    cmd.Parameters.AddWithValue("$numObsSatire", numObservedSatire.Value.ToString());
                    cmd.Parameters.AddWithValue("$obsDetectorAcc", cboSatireDetectorAcc.Text);

                    for (int c = 0; c < featureScores.Length; ++c)
                    {
                        cmd.Parameters.AddWithValue("$c" + c.ToString(), featureScores[c]);
                    }

                    cmd.Parameters.AddWithValue("$xPathTags", satireTagWnd.GetXPathTags());
                    cmd.Parameters.AddWithValue("$time", DateTime.Now.ToShortTimeString());
                    cmd.Parameters.AddWithValue("$date", DateTime.Now.ToShortDateString());

                    cmd.ExecuteNonQuery();
                    satireSaved = true;
                }
            }

            if (saveFalsifications == true)
            {
                if (txtFalsificationFeatureScores.Text != "")
                {
                    string url = txtURL.Text.Replace("'", "''");
                    string falsificationText = rtbFalsificationResults.Text;
                    falsificationText = falsificationText.Replace("'", "''");
                    string[] featureScores = txtFalsificationFeatureScores.Text.Split(',');

                    string insertFalsification = "insert into falsification values(null, $url, $falsText, $numObsNotFals, $numObsFals, $obsDetectorAcc,";

                    for (int c = 0; c < featureScores.Length; ++c)
                    {
                        insertFalsification += "$c" + Convert.ToString(c) + ", ";
                    }

                    insertFalsification += " $xPathTags, $time, $date);";

                    SQLiteCommand cmd = new SQLiteCommand(insertFalsification, conn);
                    cmd.Parameters.AddWithValue("$url", url);
                    cmd.Parameters.AddWithValue("$falsText", falsificationText);
                    cmd.Parameters.AddWithValue("$numObsNotFals", numObservedLegitScore.Value);
                    cmd.Parameters.AddWithValue("$numObsFals", numObservedFalsifiedScore.Value);
                    cmd.Parameters.AddWithValue("$obsDetectorAcc", cboFalsificationDetectorAcc.Text);

                    for (int c = 0; c < featureScores.Length; ++c)
                    {
                        cmd.Parameters.AddWithValue("$c" + c.ToString(), featureScores[c]);
                    }

                    cmd.Parameters.AddWithValue("$xPathTags", falsificationTagWnd.GetXPathTags());
                    cmd.Parameters.AddWithValue("$time", DateTime.Now.ToShortTimeString());
                    cmd.Parameters.AddWithValue("$date", DateTime.Now.ToShortDateString());

                    cmd.ExecuteNonQuery();
                    falsificationSaved = true;
                }
            }

            if (saveNotes == true)
            {
                if (rtbPageNotes.Text != "")
                {
                    string url = txtURL.Text.Replace("'", "''");
                    string insertNote = String.Format("insert into notes values(null, $url, $notes, $time, $date);");
                    SQLiteCommand cmd = new SQLiteCommand(insertNote, conn);
                    cmd.Parameters.AddWithValue("$url", url);
                    cmd.Parameters.AddWithValue("$notes", rtbPageNotes.Text);
                    cmd.Parameters.AddWithValue("$time", DateTime.Now.ToShortTimeString());
                    cmd.Parameters.AddWithValue("$date", DateTime.Now.ToShortDateString());
                    cmd.ExecuteNonQuery();
                    rtbPageNotes.Text = "";
                    noteSaved = true;
                }
            }

            conn.Close();

            string clickbaitMsg = clickbaitSaved.ToString() + " clickbait results" + Environment.NewLine;
            string satireMsg = "Page satire scores" + Environment.NewLine;
            string falsificationMsg = "Page falsification scores" + Environment.NewLine;
            string noteMsg = "Notes about this page" + Environment.NewLine;
            string saveMsg = "Saved to dataset " + CboDatabases.Items[CboDatabases.SelectedIndex];
            string fullMsg = "";

            if (clickbaitSaved > 0)
            {
                fullMsg += clickbaitMsg;
            }

            if (satireSaved)
            {
                fullMsg += satireMsg;
            }

            if (falsificationSaved)
            {
                fullMsg += falsificationMsg;
            }

            if (noteSaved)
            {
                fullMsg += noteMsg;
            }

            fullMsg += saveMsg;

            //MessageBox.Show(fullMsg);

        }

        private void BtnInsertResults_Click(object sender, EventArgs e)
        {
            SaveResultsToDB(chkClickbait.Checked, chkSatire.Checked, chkFalsification.Checked, chkNotes.Checked);
        }

        //try and remember the last dataset you selected
        private int lastSelectedDataset = -1;

        public int LastSelectedDataset { get => lastSelectedDataset; set => lastSelectedDataset = value; }

        private void TabsBrowser_Selected(object sender, TabControlEventArgs e)
        {
            if (CboDatabases.SelectedIndex != -1)
            {
                LastSelectedDataset = CboDatabases.SelectedIndex;
            }

            CboDatabases.Items.Clear();

            //copy URL from main window to the one to save to DB
            TxtPageSaveURL.Text = txtURL.Text;

            string[] datasets = Directory.GetFiles(".\\datasets");
            foreach (string fileName in datasets)
            {
                if (fileName.EndsWith(".db") == true)
                {
                    CboDatabases.Items.Add(fileName);
                }
            }

            if (LastSelectedDataset != -1)
                CboDatabases.SelectedIndex = LastSelectedDataset;
        }

        private void NumMinWordCount_ValueChanged(object sender, EventArgs e)
        {
            //NOTE: we just use this value when the page loads so we don't have tons of threads
            //FIXME: refreshes at each tick - ????
            lastURL = "";
            c.Browser.CoreWebView2.Reload();
        }

        private void BtnExploreSatire_Click(object sender, EventArgs e)
        {
            string[] featureScores = txtSatireFeatureScores.Text.Split(',');
            if (rtbSatireResults.Text == "" || featureScores.Length == 0 || featureScores[0] == "")
            {
                MessageBox.Show("No text has been selected for processing.", "Satire Detector", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                return;
            }
            frmStats.ShowSatireDetails(featureScores.ToList<string>(), rtbSatireResults.Text);
            frmStats.Show();
            frmStats.BringToFront();
        }

        //private async void BtnDetectClickbaitSelected_ClickAsync(object sender, EventArgs e)
        //{
        //    JavascriptResponse t = await c.Browser.EvaluateScriptAsync("window.getSelection().toString()");
        //    if (t.Result == null)
        //    {
        //        MessageBox.Show("Nothing was selected.", "Detector", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
        //        return;
        //    }
        //    string selection = t.Result.ToString();
        //    List<string> text = new List<string>();
        //    text.Add(selection);
        //    DetectClickbait(text);
        //}

        //private async void BtnDetectSatireSelected_ClickAsync(object sender, EventArgs e)
        //{
        //    JavascriptResponse t = await c.Browser.EvaluateScriptAsync("window.getSelection().toString()");
        //    if (t.Result == null)
        //    {
        //        MessageBox.Show("Nothing was selected.", "Detector", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
        //        return;
        //    }
        //    string selection = t.Result.ToString();
        //    rtbSatireResults.Text = selection;
        //    List<string> text = new List<string>();
        //    text.Add(selection);
        //    DetectSatire(text);
        //}

        //private async void BtnDetectFalsificationSelected_ClickAsync(object sender, EventArgs e)
        //{
        //    JavascriptResponse t = await c.Browser.EvaluateScriptAsync("window.getSelection().toString()");
        //    if (t.Result == null)
        //    {
        //        MessageBox.Show("Nothing was selected.", "Detector", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
        //        return;
        //    }
        //    string selection = t.Result.ToString();
        //    rtbFalsificationResults.Text = selection;
        //    List<string> text = new List<string>();
        //    text.Add(selection);
        //    DetectFalsification(text);
        //}

        private void BtnExploreFalsification_Click(object sender, EventArgs e)
        {
            string[] featureScores = txtFalsificationFeatureScores.Text.Split(',');
            if (rtbSatireResults.Text == "" || featureScores.Length == 0 || featureScores[0] == "")
            {
                MessageBox.Show("No text has been selected for processing.", "Falsification Detector", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                return;
            }
            frmStats.ShowFalsificationDetails(featureScores.ToList<string>(), rtbFalsificationResults.Text);
            frmStats.Show();
            frmStats.BringToFront();
        }

        private FrmSelectHtmlTags clickbaitTagWnd;
        private void BtnTagSelectClickbait_Click(object sender, EventArgs e)
        {
            clickbaitTagWnd.ShowDialog();
            lastURL = "";   //need to clear so refreshes work
            //c.Browser.GetBrowser().Reload();
        }

        private FrmSelectHtmlTags satireTagWnd;
        private void BtnTagSelectSatire_Click(object sender, EventArgs e)
        {
            satireTagWnd.ShowSatireCharLength();
            satireTagWnd.ShowDialog();

            lastURL = "";   //need to clear so refreshes work
            //c.Browser.GetBrowser().Reload();
        }

        private FrmSelectHtmlTags falsificationTagWnd;

        private void BtnTagSelectFalsification_Click(object sender, EventArgs e)
        {
            falsificationTagWnd.ShowFalsCharLength();
            falsificationTagWnd.ShowDialog();

            lastURL = "";   //need to clear so refreshes work
            //c.Browser.GetBrowser().Reload();
        }

        private void BtnOpenLocalPage_Click(object sender, EventArgs e)
        {
            OpenFileDialog openPageDialog = new OpenFileDialog();
            openPageDialog.Filter = "HTML Files (*.html)|*.html|HTML Files (*.htm)|*.htm|Text files (*.txt)|*.txt";
            openPageDialog.Title = "Select a file";
            if (openPageDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                string file = File.ReadAllText(openPageDialog.FileName);
                //c.Browser.CoreWebView2.LoadHtml(file, openPageDialog.FileName);
            }
        }

        private void BtnSavePageToDisk_Click(object sender, EventArgs e)
        {
            throw new NotImplementedException();
        }

        private async void BtnInputSendToBrowser_ClickAsync(object sender, EventArgs e)
        {
            //string x = await editorInput.Browser.GetSourceAsync();
            //HtmlAgilityPack.HtmlDocument htm = new HtmlAgilityPack.HtmlDocument();
            //htm.LoadHtml(x);
            //HtmlNode bodyNode = htm.DocumentNode.SelectSingleNode("//body");
            //bodyNode.Attributes.Remove();
            //lastURL = "";   //need to clear so refreshes work
            //c.Browser.LoadHtml(htm.DocumentNode.OuterHtml, "http://yourinputdoc");
            //TabsBrowser.SelectTab(0);
        }

        private void BtnInputClear_Click(object sender, EventArgs e)
        {
            //editorInput.Browser.LoadHtml("<html><body contentEditable='true'></body></html>", "http://yourinputdoc");
        }

        private void BtnInputBold_ClickAsync(object sender, EventArgs e)
        {
            editorInput.Browser.ExecuteScriptAsync("document.execCommand('bold')");
        }

        private void BtnInputH1_Click(object sender, EventArgs e)
        {
            editorInput.Browser.ExecuteScriptAsync("document.execCommand('formatBlock', false, 'h1')");
        }

        private void btnInputH2_Click(object sender, EventArgs e)
        {
            editorInput.Browser.ExecuteScriptAsync("document.execCommand('formatBlock', false, 'h2')");
        }

        private void BtnInputH3_Click(object sender, EventArgs e)
        {
            editorInput.Browser.ExecuteScriptAsync("document.execCommand('formatBlock', false, 'h3')");
        }

        private void BtnInputH4_Click(object sender, EventArgs e)
        {
            editorInput.Browser.ExecuteScriptAsync("document.execCommand('formatBlock', false, 'h4')");
        }

        private void BtnInputH5_Click(object sender, EventArgs e)
        {
            editorInput.Browser.ExecuteScriptAsync("document.execCommand('formatBlock', false, 'h5')");
        }

        private void BtnInputP_Click(object sender, EventArgs e)
        {
            editorInput.Browser.ExecuteScriptAsync("document.execCommand('formatBlock', false, 'p')");
        }

        private void BtnInputU_Click(object sender, EventArgs e)
        {
            editorInput.Browser.ExecuteScriptAsync("document.execCommand('underline')");
        }

        delegate void ClickbaitHighlightCallBack();
        private void ClickbaitHighlight()
        {
            if (TabsDetectors.SelectedTab.Text != "Clickbait" && ChkAlwaysProcessClickbait.Checked == false)
                return;

            //note that ".replace(/[“’”]/g, '')" was added to compensate for the encoding issues between the UI and Python 2.7... can be removed when upgraded to 3

            string jsClickbait = @"
                    var notClickbait = [
                        {0}
                    ];
                    var slightClickbait = [
                        {1}
                    ];
                    var moderateClickbait = [
                        {2}
                    ];
                    var heavyClickbait = [
                        {3}
                    ];

                    var lnks = [];

                    {4}

                    console.log(lnks);

                    var notClickbaitIndex = 0;
                    var slightClickbaitIndex = 0;
                    var moderateClickbaitIndex = 0;
                    var heavyClickbaitIndex = 0;

                    console.log(notClickbait);
                    console.log(slightClickbait);
                    console.log(moderateClickbait);
                    console.log(heavyClickbait);
                    console.log(lnks);
                    console.log(lnks.length);

                    for (var i = notClickbait.length - 2; i < notClickbait.length; ++i) {{
                        for (var j = 0; j < lnks.length; ++j) {{
                            if (lnks[j].innerText.replace(/[“’”—ó‘’á]/g, '') == notClickbait[i]) {{
                                console.log('match: ' + lnks[j].innerText + ' ' + notClickbait[i]);
                                lnks[j].style.backgroundColor = 'rgba({5}, 0.7)';
                            }}
                        }}
                    }}
                    for (var i = slightClickbait.length - 2; i < slightClickbait.length; ++i) {{
                        for (var j = 0; j < lnks.length; ++j) {{
                            if (lnks[j].innerText.replace(/[“’”—ó‘’á]/g, '')  == slightClickbait[i]) {{
                                console.log('match: ' + lnks[j].innerText + ' ' + slightClickbait[i]);
                                lnks[j].style.backgroundColor = 'rgba({6}, 0.7)';;
                            }}
                        }}
                    }}
                    for (var i = moderateClickbait.length - 2; i < moderateClickbait.length; ++i) {{
                        for (var j = 0; j < lnks.length; ++j) {{
                            if (lnks[j].innerText.replace(/[“’”—ó‘’á]/g, '')  == moderateClickbait[i]) {{
                                console.log('match: ' + lnks[j].innerText + ' ' + moderateClickbait[i]);
                                lnks[j].style.backgroundColor = 'rgba({7}, 0.7)';
                            }}
                        }}
                    }}
                    for (var i = heavyClickbait.length - 2; i < heavyClickbait.length; ++i) {{
                        for (var j = 0; j < lnks.length; ++j) {{
                            if (lnks[j].innerText.replace(/[“’”—ó‘’á]/g, '') == heavyClickbait[i]) {{
                                console.log('match: ' + lnks[j].innerText + ' ' + heavyClickbait[i]);
                                lnks[j].style.backgroundColor = 'rgba({8}, 0.7)';
                            }}
                        }}
                    }}";

            string js = "";
            string tags = "";

            string notClickbaitString = string.Join("\",\"", ncb);
            string slightClickbaitString = string.Join("\",\"", scb);
            string moderateClickbaitString = string.Join("\",\"", mcb);
            string heavyClickbaitString = string.Join("\",\"", hcb);
            tags = clickbaitTagWnd.GetJSTags("lnks.push.apply(lnks, document.getElementsByTagName('", "'));");

            if (!String.IsNullOrEmpty(notClickbaitString))
            {
                notClickbaitString = "\"" + notClickbaitString + "\"";
            }
            if (!String.IsNullOrEmpty(slightClickbaitString))
            {
                slightClickbaitString = "\"" + slightClickbaitString + "\"";
            }
            if (!String.IsNullOrEmpty(moderateClickbaitString))
            {
                moderateClickbaitString = "\"" + moderateClickbaitString + "\"";
            }
            if (!String.IsNullOrEmpty(heavyClickbaitString))
            {
                heavyClickbaitString = "\"" + heavyClickbaitString + "\"";
            }

            js = String.Format(jsClickbait, notClickbaitString, slightClickbaitString, moderateClickbaitString, heavyClickbaitString, tags,
                CLICKBAIT_LOW_COLOR, CLICKBAIT_SLIGHT_COLOR, CLICKBAIT_MODERATE_COLOR, CLICKBAIT_HEAVY_COLOR);

            c.Browser.ExecuteScriptAsync(js);
        }

        delegate void SatireHighlightCallBack();
        private void SatireHighlight()
        {
            if (TabsDetectors.SelectedTab.Text != "Satire")
                return;

            string jsHighlight = @"
                var texts = [];

                {0}

                for (var j = 0; j < texts.length; ++j) {{
                    texts[j].style.backgroundColor = 'rgba({1}, 0.5)';
                }}";

            string js = "";
            string tags = "";
            string color = "";

            tags = satireTagWnd.GetJSTags("texts.push.apply(texts, document.getElementsByTagName('", "'));");

            if (SATIRE_MOSTLY_LEGIT)
                color = NOT_SATIRE_TEXT_COLOR;
            else
                color = SATIRE_TEXT_COLOR;

            js = String.Format(jsHighlight, tags, color);

            c.Browser.ExecuteScriptAsync(js);
        }

        delegate void FalsificationHighlightCallBack();
        private void FalsificationHighlight()
        {
            if (TabsDetectors.SelectedTab.Text != "Falsifications")
                return;

            string jsHighlight = @"
                var texts = [];

                {0}

                for (var j = 0; j < texts.length; ++j) {{
                    texts[j].style.backgroundColor = 'rgba({1}, 0.5)';
                }}";

            string js = "";
            string tags = "";
            string color = "";

            tags = falsificationTagWnd.GetJSTags("texts.push.apply(texts, document.getElementsByTagName('", "'));");

            if (FALSIFICATION_MOSTLY_LEGIT)
                color = NOT_FALSIFICATION_TEXT_COLOR;
            else
                color = FALSIFICATION_TEXT_COLOR;

            js = String.Format(jsHighlight, tags, color);

            c.Browser.ExecuteScriptAsync(js);
        }

        private void SwitchHighlightingContext()
        {
            string jsHighlight = @"
                var texts = [];

                {0}

                for (var j = 0; j < texts.length; ++j) {{
                    texts[j].style.backgroundColor = 'rgba({1}, 0.5)';
                }}";

            string jsClear = @"
                var texts = [];

                {0}

                for (var j = 0; j < texts.length; ++j) {{
                    texts[j].style.backgroundColor = '';
                }}";

            string js = "";
            string tags = "";
            string color = "";

            //clear the highlighting
            if (TabsDetectors.SelectedTab.Text != "CLICKBAIT" || ChkAlwaysProcessClickbait.Checked == true)
            {
                tags = clickbaitTagWnd.GetJSTags("texts.push.apply(texts, document.getElementsByTagName('", "'));");
                js = String.Format(jsClear, tags);
                c.Browser.ExecuteScriptAsync(js);
            }
            tags = satireTagWnd.GetJSTags("texts.push.apply(texts, document.getElementsByTagName('", "'));");
            js = String.Format(jsClear, tags);
            c.Browser.ExecuteScriptAsync(js);
            tags = falsificationTagWnd.GetJSTags("texts.push.apply(texts, document.getElementsByTagName('", "'));");
            js = String.Format(jsClear, tags);
            c.Browser.ExecuteScriptAsync(js);

            if (TabsDetectors.SelectedTab.Text == "CLICKBAIT" || ChkAlwaysProcessClickbait.Checked == true)
            {
                string jsClickbait = @"
                    var notClickbait = [
                        {0}
                    ];
                    var slightClickbait = [
                        {1}
                    ];
                    var moderateClickbait = [
                        {2}
                    ];
                    var heavyClickbait = [
                        {3}
                    ];

                    var lnks = [];

                    {4}

                    console.log(lnks);

                    var notClickbaitIndex = 0;
                    var slightClickbaitIndex = 0;
                    var moderateClickbaitIndex = 0;
                    var heavyClickbaitIndex = 0;

                    console.log(notClickbait);
                    console.log(slightClickbait);
                    console.log(moderateClickbait);
                    console.log(heavyClickbait);
                    console.log(lnks);
                    console.log(lnks.length);

                    for (var i = 0; i < notClickbait.length; ++i) {{
                        for (var j = 0; j < lnks.length; ++j) {{
                            if (lnks[j].innerText.replace(/[“’”—ó‘’á]/g, '') == notClickbait[i]) {{
                                console.log('match: ' + lnks[j].innerText + ' ' + notClickbait[i]);
                                lnks[j].style.backgroundColor = 'rgba({5}, 0.7)';
                            }}
                        }}
                    }}
                    for (var i = 0; i < slightClickbait.length; ++i) {{
                        for (var j = 0; j < lnks.length; ++j) {{
                            if (lnks[j].innerText.replace(/[“’”—ó‘’á]/g, '')  == slightClickbait[i]) {{
                                console.log('match: ' + lnks[j].innerText + ' ' + slightClickbait[i]);
                                lnks[j].style.backgroundColor = 'rgba({6}, 0.7)';;
                            }}
                        }}
                    }}
                    for (var i = 0; i < moderateClickbait.length; ++i) {{
                        for (var j = 0; j < lnks.length; ++j) {{
                            if (lnks[j].innerText.replace(/[“’”—ó‘’á]/g, '')  == moderateClickbait[i]) {{
                                console.log('match: ' + lnks[j].innerText + ' ' + moderateClickbait[i]);
                                lnks[j].style.backgroundColor = 'rgba({7}, 0.7)';
                            }}
                        }}
                    }}
                    for (var i = 0; i < heavyClickbait.length; ++i) {{
                        for (var j = 0; j < lnks.length; ++j) {{
                            if (lnks[j].innerText.replace(/[“’”—ó‘’á]/g, '') == heavyClickbait[i]) {{
                                console.log('match: ' + lnks[j].innerText + ' ' + heavyClickbait[i]);
                                lnks[j].style.backgroundColor = 'rgba({8}, 0.7)';
                            }}
                        }}
                    }}";

                string notClickbaitString = string.Join("\",\"", ncb);
                string slightClickbaitString = string.Join("\",\"", scb);
                string moderateClickbaitString = string.Join("\",\"", mcb);
                string heavyClickbaitString = string.Join("\",\"", hcb);
                tags = clickbaitTagWnd.GetJSTags("lnks.push.apply(lnks, document.getElementsByTagName('", "'));");

                if (!String.IsNullOrEmpty(notClickbaitString))
                {
                    notClickbaitString = "\"" + notClickbaitString + "\"";
                }
                if (!String.IsNullOrEmpty(slightClickbaitString))
                {
                    slightClickbaitString = "\"" + slightClickbaitString + "\"";
                }
                if (!String.IsNullOrEmpty(moderateClickbaitString))
                {
                    moderateClickbaitString = "\"" + moderateClickbaitString + "\"";
                }
                if (!String.IsNullOrEmpty(heavyClickbaitString))
                {
                    heavyClickbaitString = "\"" + heavyClickbaitString + "\"";
                }

                js = String.Format(jsClickbait, notClickbaitString, slightClickbaitString, moderateClickbaitString, heavyClickbaitString, tags,
                    CLICKBAIT_LOW_COLOR, CLICKBAIT_SLIGHT_COLOR, CLICKBAIT_MODERATE_COLOR, CLICKBAIT_HEAVY_COLOR);

                c.Browser.ExecuteScriptAsync(js);
            }

            if (TabsDetectors.SelectedTab.Text == "SATIRE")
            {
                tags = satireTagWnd.GetJSTags("texts.push.apply(texts, document.getElementsByTagName('", "'));");

                if (SATIRE_MOSTLY_LEGIT)
                    color = NOT_SATIRE_TEXT_COLOR;
                else
                    color = SATIRE_TEXT_COLOR;

                js = String.Format(jsHighlight, tags, color);

                c.Browser.ExecuteScriptAsync(js);
            }
            else if (TabsDetectors.SelectedTab.Text == "FALSIFICATIONS (ALPHA)")
            {
                tags = falsificationTagWnd.GetJSTags("texts.push.apply(texts, document.getElementsByTagName('", "'));");

                if (FALSIFICATION_MOSTLY_LEGIT)
                    color = NOT_FALSIFICATION_TEXT_COLOR;
                else
                    color = FALSIFICATION_TEXT_COLOR;

                js = String.Format(jsHighlight, tags, color);

                c.Browser.ExecuteScriptAsync(js);
            }
        }

        private void TabsDetectors_SelectedIndexChanged(object sender, EventArgs e)
        {
            SwitchHighlightingContext();

            TxtWaybackURL.Text = txtURL.Text;
        }

        private void TabsBrowser_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (TabsBrowser.SelectedTab.Text == "ANALYSIS")
            {
                splitContainer1.Panel2Collapsed = true;
                UpdateBookmarks();
            }
            else if (TabsBrowser.SelectedTab.Text == "INPUT")
            {
                splitContainer1.Panel2Collapsed = true;
            }
            else
            {
                splitContainer1.Panel2Collapsed = false;
            }
        }

        private void ReadHighlightColors()
        {
            ReadClickbaitColors();
            ReadSatireColors();
            ReadFalsificationColors();
        }

        private void ReadClickbaitColors()
        {
            SQLiteConnection conn = new SQLiteConnection("Data Source=bookmarks.db;Version=3");
            conn.Open();
            SQLiteCommand getColors = new SQLiteCommand("select * from colors_clickbait;", conn);
            SQLiteDataReader readColors = getColors.ExecuteReader();
            while (readColors.Read())
            {
                CLICKBAIT_LOW_COLOR = readColors["cb_none"].ToString();
                CLICKBAIT_MODERATE_COLOR = readColors["cb_moderate"].ToString();
                CLICKBAIT_SLIGHT_COLOR = readColors["cb_slight"].ToString();
                CLICKBAIT_HEAVY_COLOR = readColors["cb_heavy"].ToString();
            }
            conn.Close();
        }

        private void ReadSatireColors()
        {
            SQLiteConnection conn = new SQLiteConnection("Data Source=bookmarks.db;Version=3");
            conn.Open();
            SQLiteCommand getColors = new SQLiteCommand("select * from colors_satire;", conn);
            SQLiteDataReader readColors = getColors.ExecuteReader();
            while (readColors.Read())
            {
                SATIRE_TEXT_COLOR = readColors["satirical"].ToString();
                NOT_SATIRE_TEXT_COLOR = readColors["notsatirical"].ToString();
            }
            conn.Close();
            LblDetectorNotSatire.ForeColor = FrmColorSettings.GetColorFromString(NOT_SATIRE_TEXT_COLOR);
            LblDetectorSatire.ForeColor = FrmColorSettings.GetColorFromString(SATIRE_TEXT_COLOR);
        }

        private void ReadFalsificationColors()
        {
            SQLiteConnection conn = new SQLiteConnection("Data Source=bookmarks.db;Version=3");
            conn.Open();
            SQLiteCommand getColors = new SQLiteCommand("select * from colors_falsifications;", conn);
            SQLiteDataReader readColors = getColors.ExecuteReader();
            while (readColors.Read())
            {
                FALSIFICATION_TEXT_COLOR = readColors["falsification"].ToString();
                NOT_FALSIFICATION_TEXT_COLOR = readColors["notfalsification"].ToString();
            }
            conn.Close();
            LblDetectorFalsified.ForeColor = FrmColorSettings.GetColorFromString(FALSIFICATION_TEXT_COLOR);
            LblDetectorLegit.ForeColor = FrmColorSettings.GetColorFromString(NOT_FALSIFICATION_TEXT_COLOR);
        }

        private void UpdateBookmarks()
        {
            LstBookmarks.Items.Clear();
            SQLiteConnection conn = new SQLiteConnection("Data Source=bookmarks.db;Version=3");
            conn.Open();
            SQLiteCommand getBookmarks = new SQLiteCommand("select url from bookmarks;", conn);
            SQLiteDataReader readBookmarks = getBookmarks.ExecuteReader();
            while (readBookmarks.Read())
            {
                LstBookmarks.Items.Add(readBookmarks["url"]);
            }
            conn.Close();
        }

        private void BtnAddBookmark_Click(object sender, EventArgs e)
        {
            SQLiteConnection conn = new SQLiteConnection("Data Source=bookmarks.db;Version=3");
            conn.Open();

            string insertURL = String.Format("insert into bookmarks values('{0}');", txtURL.Text);

            SQLiteCommand cmd = new SQLiteCommand(insertURL, conn);
            cmd.ExecuteNonQuery();
            conn.Close();
            UpdateBookmarks();
        }

        private void BtnDeleteBookmark_Click(object sender, EventArgs e)
        {
            SQLiteConnection conn = new SQLiteConnection("Data Source=bookmarks.db;Version=3");
            conn.Open();

            string deleteURL = String.Format("delete from bookmarks where url = '{0}';", LstBookmarks.SelectedItem.ToString());

            SQLiteCommand cmd = new SQLiteCommand(deleteURL, conn);
            cmd.ExecuteNonQuery();
            conn.Close();
            UpdateBookmarks();
        }

        private void TxtHomepageURL_TextChanged(object sender, EventArgs e)
        {
            File.WriteAllText("homepage.path", TxtHomepageURL.Text);
        }

        private void BtnSetHomepage_Click(object sender, EventArgs e)
        {
            TxtHomepageURL.Text = txtURL.Text;
        }

        private void BtnHome_Click(object sender, EventArgs e)
        {
            lastURL = "";
            //c.Browser.Load(TxtHomepageURL.Text);
        }

        private void LstBookmarks_SelectedValueChanged(object sender, EventArgs e)
        {
            if (LstBookmarks.SelectedItems.Count > 0)
            {
                lastURL = "";
                //c.Browser.Load(LstBookmarks.SelectedItems[0].ToString());
            }
        }

        private void ShowHighlightColorDialog()
        {
            FrmColorSettings f = new FrmColorSettings();
            f.ShowDialog();
            f.Close();
            ReadHighlightColors();
        }

        private void BtnClickbaitColors_Click(object sender, EventArgs e)
        {
            ShowHighlightColorDialog();
        }

        private void BtnSatireColors_Click(object sender, EventArgs e)
        {
            ShowHighlightColorDialog();
        }

        private void BtnFalsColors_Click(object sender, EventArgs e)
        {
            ShowHighlightColorDialog();
        }

        private void TxtCBFilter_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Enter)
            {
                lastURL = "";
                c.Navigate(txtURL.Text);
                e.SuppressKeyPress = true;
                e.Handled = true;
            }
        }

        private void FrmMain_Shown(object sender, EventArgs e)
        {
            MessageBox.Show("The Web Research Manager is experimental research software and is not perfect and is not correct all the time. Digital literacy is key for everyone to effectively evaluate potential misinformation online, and this program is NOT a replacement for that.  Use this program with caution!", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void CboWaybackYears_SelectedIndexChanged(object sender, EventArgs e)
        {
            DateTime selectedYear = new DateTime(Convert.ToInt32(CboWaybackYears.SelectedItem.ToString()), 1, 1);
            monthCalendarWayback.SetDate(selectedYear);
        }

        private async void BtnCheckWayback_Click(object sender, EventArgs e)
        {
            DateTime selectedDay = monthCalendarWayback.SelectionStart;

            string waybackUrl = "http://archive.org/wayback/available?url=";
            string timestamp = "&timestamp=";
            string fullUrl = waybackUrl + TxtWaybackURL.Text + timestamp + selectedDay.Year.ToString() + selectedDay.Month.ToString() + selectedDay.Day.ToString();

            HttpClient client = new HttpClient();
            client.DefaultRequestHeaders.Accept.Clear();
            var response = client.GetStringAsync(fullUrl);
            await response;

            string html = response.Result.ToString();

            if (html == "{\"archived_snapshots\":{}}")
            {
                LblFoundWaybackDesc.Text = "No wayback page found.";
                PanWaybackCheckOutput.Visible = true;
                LnkLblURLForWayback.Visible = false;
                LnkLblOriginalForWayback.Visible = false;
                LblWaybackClosestTimeDiff.Visible = false;
            }
            else
            {
                dynamic json = JsonConvert.DeserializeObject<dynamic>(html);
                // MessageBox.Show(json[1][1].ToString());

                string status = json.archived_snapshots.closest.status;
                string available = json.archived_snapshots.closest.available;
                string ts = json.archived_snapshots.closest.timestamp;
                string snapshoturl = json.archived_snapshots.closest.url;

                if (status == "200" && available == "True")
                {
                    LblFoundWaybackDesc.Text = "Found wayback on " + ts;
                    LnkLblURLForWayback.Text = snapshoturl;
                    LnkLblOriginalForWayback.Text = TxtWaybackURL.Text;

                    string year = ts.Substring(0, 4);
                    string mm = ts.Substring(4, 2);
                    string dd = ts.Substring(6, 2);

                    DateTime closestDt = new DateTime(Convert.ToInt32(year), Convert.ToInt32(mm), Convert.ToInt32(dd));
                    TimeSpan result;

                    if (selectedDay > closestDt)
                    {
                        result = selectedDay.Subtract(closestDt);
                    }
                    else
                    {
                        result = closestDt.Subtract(selectedDay);
                    }

                    LblWaybackClosestTimeDiff.Text = "Closest Difference (days) : " + result.Days.ToString();

                    PanWaybackCheckOutput.Visible = true;
                    LnkLblURLForWayback.Visible = true;
                    LnkLblOriginalForWayback.Visible = true;
                    LblWaybackClosestTimeDiff.Visible = true;
                }
                else
                {
                    MessageBox.Show("Unexpected Result from Wayback.");
                }
            }
        }

        private void BtnSendDatesToCrawler_Click(object sender, EventArgs e)
        {

        }

        private void BtnSetCurrentURLWaybackCheck_Click(object sender, EventArgs e)
        {
            TxtWaybackURL.Text = txtURL.Text;
        }

        private void LnkLblURLForWayback_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            c.Navigate(LnkLblURLForWayback.Text);
        }

        private void LnkLblOriginalForWayback_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            c.Navigate(LnkLblOriginalForWayback.Text);
        }

        private void BtnSaveCrawlOutput_Click(object sender, EventArgs e)
        {

        }

        private void BtnCrawlPause_Click(object sender, EventArgs e)
        {

        }

        private bool PauseCrawl = false;
        private List<string> crawlURLs = new List<string>();
        private int crawlIndex = 0;
        private bool CrawlMode = false;

        private async void BtnSetCrawlMode_Click(object sender, EventArgs e)
        {
            if (CboDatabases.SelectedIndex == -1)
            {
                MessageBox.Show("Please select a database under Document Analysis to begin crawl.");
                return;
            }

            if (CboCrawlFrequency.SelectedIndex == -1)
            {
                MessageBox.Show("Please select a Crawl Frequency.");
                return;
            }

            if (CboCrawlMode.SelectedIndex == -1)
            {
                MessageBox.Show("Please select a Crawl Mode.");
                return;
            }

            if (CboCrawlMode.SelectedText == "Wayback Only" || CboCrawlMode.SelectedText == "Both Current and Wayback")
            {
                if (TxtCrawlWaybackPrefDate.Text == "")
                {
                    MessageBox.Show("For Wayback crawls, please select a Wayback date under the Wayback tab.");
                    return;
                }
            }

            if (RtbCrawlLinkList.Text == "")
            {
                MessageBox.Show("Please input at least one URL in the URLs textbox. For more than one, use line-by-line format.");
                return;
            }

            if (LstCrawlSaveResults.SelectedItems.Count == 0)
            {
                MessageBox.Show("Please select at least one measure to save.");
                return;
            }


            CrawlMode = true;
            BtnSetCrawlMode.Text = "Crawl Mode is ON";
            BtnSetCrawlMode.BackColor = Color.DarkGreen;

            TxtCrawlOutput.Text = "[" + DateTime.Now.ToString() + "] " + "Checking for Wayback pages." + Environment.NewLine;
            crawlURLs = new List<string>(RtbCrawlLinkList.Lines);
            RtbCrawlLinkList.ReadOnly = true;

            List<string> allCrawlURLs = new List<string>();
            foreach (string url in crawlURLs)
            {
                if (CboCrawlMode.SelectedText == "Current Page Only")
                {
                    allCrawlURLs.Add(url);
                    continue;
                }
                else if (CboCrawlMode.SelectedText == "Wayback Only")
                {
                    ;
                }
                else if (CboCrawlMode.SelectedText == "Both Current and Wayback")
                {
                    allCrawlURLs.Add(url);
                }

                DateTime selectedDay = monthCalendarWayback.SelectionStart;

                string waybackUrl = "http://archive.org/wayback/available?url=";
                string timestamp = "&timestamp=";
                string fullUrl = waybackUrl + url + timestamp + selectedDay.Year.ToString() + selectedDay.Month.ToString() + selectedDay.Day.ToString();

                HttpClient client = new HttpClient();
                client.DefaultRequestHeaders.Accept.Clear();
                var response = client.GetStringAsync(fullUrl);
                await response;

                string html = response.Result.ToString();

                try
                {
                    dynamic json = JsonConvert.DeserializeObject<dynamic>(html);
                    // MessageBox.Show(json[1][1].ToString());

                    string status = json.archived_snapshots.closest.status;
                    string available = json.archived_snapshots.closest.available;
                    string ts = json.archived_snapshots.closest.timestamp;
                    string snapshoturl = json.archived_snapshots.closest.url;

                    if (status == "200" && available == "True")
                    {
                        TxtCrawlOutput.Text += "[" + DateTime.Now.ToString() + "] " + "Found wayback for " + TxtWaybackURL.Text + " on " + ts + " at " + snapshoturl + Environment.NewLine;

                        string year = ts.Substring(0, 4);
                        string mm = ts.Substring(4, 2);
                        string dd = ts.Substring(6, 2);

                        DateTime closestDt = new DateTime(Convert.ToInt32(year), Convert.ToInt32(mm), Convert.ToInt32(dd));
                        TimeSpan result;

                        if (selectedDay > closestDt)
                        {
                            result = selectedDay.Subtract(closestDt);
                        }
                        else
                        {
                            result = closestDt.Subtract(selectedDay);
                        }

                        TxtCrawlOutput.Text += "Closest Difference (days) : " + result.Days.ToString() + Environment.NewLine;

                        allCrawlURLs.Add(snapshoturl);
                    }
                    else
                    {
                        TxtCrawlOutput.Text += "Unexpected Result from Wayback: " + url + Environment.NewLine;
                        continue;
                    }
                }
                catch (Exception ex)
                {
                    TxtCrawlOutput.Text += "Unexpected Result from Wayback: " + url + Environment.NewLine;
                }


            }
            RtbCrawlLinkList.ReadOnly = false;
            RtbCrawlLinkList.Lines = allCrawlURLs.ToArray<string>();
            RtbCrawlLinkList.ReadOnly = true;
            crawlURLs = allCrawlURLs;

            c.Navigate(crawlURLs[crawlIndex]);
            TxtCrawlOutput.Text += "[" + DateTime.Now.ToString() + "] " + "Starting crawls on " + crawlURLs[crawlIndex] + Environment.NewLine;
        }

        private void monthCalendarWayback_DateChanged(object sender, DateRangeEventArgs e)
        {
            DateTime selectedDay = monthCalendarWayback.SelectionStart;
            string fullUrl = selectedDay.Year.ToString() + selectedDay.Month.ToString() + selectedDay.Day.ToString();
            TxtCrawlWaybackPrefDate.Text = fullUrl;
        }

    }
}
