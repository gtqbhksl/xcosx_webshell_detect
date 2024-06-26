package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/patrikeh/go-deep"
	"io/fs"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"wxel/core"
)

const (
	MaxFileSize   = 10 * 1024 * 1024
	EndSig        = "__WXX__"
	ModuleContent = `
{
  "Layers": [
    {
      "Neurons": [
        {
          "In": [
            {
              "Weight": 0.07802227838030487,
              "IsBias": false
            },
            {
              "Weight": 1.4095429709623049,
              "IsBias": false
            },
            {
              "Weight": 0.38148619141265666,
              "IsBias": false
            },
            {
              "Weight": -6.362409388325939,
              "IsBias": false
            },
            {
              "Weight": 4.080571054281222,
              "IsBias": false
            },
            {
              "Weight": -1.107085408804461,
              "IsBias": false
            },
            {
              "Weight": -2.807126016126731,
              "IsBias": false
            },
            {
              "Weight": -1.9763170163830057,
              "IsBias": true
            }
          ],
          "Out": [
            {
              "Weight": 2.823264326167646,
              "IsBias": false
            },
            {
              "Weight": -0.6608600824674566,
              "IsBias": false
            },
            {
              "Weight": 2.874326287374728,
              "IsBias": false
            },
            {
              "Weight": 0.08416886315759835,
              "IsBias": false
            },
            {
              "Weight": 0.8153010040531086,
              "IsBias": false
            },
            {
              "Weight": -4.472337116575626,
              "IsBias": false
            },
            {
              "Weight": 0.541617145636806,
              "IsBias": false
            }
          ]
        },
        {
          "In": [
            {
              "Weight": -0.6543721724426969,
              "IsBias": false
            },
            {
              "Weight": -0.11351002316301177,
              "IsBias": false
            },
            {
              "Weight": -2.778411050495509,
              "IsBias": false
            },
            {
              "Weight": -1.5991736481774113,
              "IsBias": false
            },
            {
              "Weight": 0.9136739867878497,
              "IsBias": false
            },
            {
              "Weight": -0.644907220299406,
              "IsBias": false
            },
            {
              "Weight": -0.8640656179897181,
              "IsBias": false
            },
            {
              "Weight": -0.9315779426960278,
              "IsBias": true
            }
          ],
          "Out": [
            {
              "Weight": -1.4567022684114097,
              "IsBias": false
            },
            {
              "Weight": -0.6468592746969144,
              "IsBias": false
            },
            {
              "Weight": 0.6201992476017124,
              "IsBias": false
            },
            {
              "Weight": 0.9125011159360927,
              "IsBias": false
            },
            {
              "Weight": 2.607038733396296,
              "IsBias": false
            },
            {
              "Weight": -1.0749890159529927,
              "IsBias": false
            },
            {
              "Weight": -1.0208558454295986,
              "IsBias": false
            }
          ]
        },
        {
          "In": [
            {
              "Weight": -1.1872150471109126,
              "IsBias": false
            },
            {
              "Weight": -7.463649545207768,
              "IsBias": false
            },
            {
              "Weight": 2.541632683678553,
              "IsBias": false
            },
            {
              "Weight": 4.160943382846905,
              "IsBias": false
            },
            {
              "Weight": -2.5973806466094373,
              "IsBias": false
            },
            {
              "Weight": 1.653957746597271,
              "IsBias": false
            },
            {
              "Weight": -6.884847050043971,
              "IsBias": false
            },
            {
              "Weight": -0.2102350505281358,
              "IsBias": true
            }
          ],
          "Out": [
            {
              "Weight": 2.554024565257,
              "IsBias": false
            },
            {
              "Weight": -8.859300727382568,
              "IsBias": false
            },
            {
              "Weight": 2.124795830106873,
              "IsBias": false
            },
            {
              "Weight": -1.9398906019689914,
              "IsBias": false
            },
            {
              "Weight": -0.4523080851775894,
              "IsBias": false
            },
            {
              "Weight": 1.9595764594365557,
              "IsBias": false
            },
            {
              "Weight": 1.5940597677757709,
              "IsBias": false
            }
          ]
        },
        {
          "In": [
            {
              "Weight": -0.8161856401698268,
              "IsBias": false
            },
            {
              "Weight": 0.3635586661044913,
              "IsBias": false
            },
            {
              "Weight": -0.8561591068732642,
              "IsBias": false
            },
            {
              "Weight": 3.2480544414176213,
              "IsBias": false
            },
            {
              "Weight": 2.7176779107757576,
              "IsBias": false
            },
            {
              "Weight": 2.114108264406231,
              "IsBias": false
            },
            {
              "Weight": -2.7332781368465193,
              "IsBias": false
            },
            {
              "Weight": -1.7746029214858727,
              "IsBias": true
            }
          ],
          "Out": [
            {
              "Weight": 0.4858741702633786,
              "IsBias": false
            },
            {
              "Weight": -1.9626762022939586,
              "IsBias": false
            },
            {
              "Weight": 1.564911000482225,
              "IsBias": false
            },
            {
              "Weight": -0.5474493229412074,
              "IsBias": false
            },
            {
              "Weight": 0.9770208354067265,
              "IsBias": false
            },
            {
              "Weight": 5.264125280855112,
              "IsBias": false
            },
            {
              "Weight": 0.8485257255411578,
              "IsBias": false
            }
          ]
        },
        {
          "In": [
            {
              "Weight": 1.5126562493704359,
              "IsBias": false
            },
            {
              "Weight": 1.163872021778959,
              "IsBias": false
            },
            {
              "Weight": -1.383147543863053,
              "IsBias": false
            },
            {
              "Weight": -0.9361762294091227,
              "IsBias": false
            },
            {
              "Weight": -3.427184157113483,
              "IsBias": false
            },
            {
              "Weight": 1.0491574601330103,
              "IsBias": false
            },
            {
              "Weight": 2.888658925819805,
              "IsBias": false
            },
            {
              "Weight": -0.2488760831751973,
              "IsBias": true
            }
          ],
          "Out": [
            {
              "Weight": 0.42286271187185837,
              "IsBias": false
            },
            {
              "Weight": 4.420592558849064,
              "IsBias": false
            },
            {
              "Weight": -0.005199491406042553,
              "IsBias": false
            },
            {
              "Weight": -1.8270924471744798,
              "IsBias": false
            },
            {
              "Weight": -1.8412769059668865,
              "IsBias": false
            },
            {
              "Weight": -0.0002618591164747282,
              "IsBias": false
            },
            {
              "Weight": 1.652621845539065,
              "IsBias": false
            }
          ]
        },
        {
          "In": [
            {
              "Weight": 0.1508130089938433,
              "IsBias": false
            },
            {
              "Weight": -23.046630100513514,
              "IsBias": false
            },
            {
              "Weight": 6.5465449241651905,
              "IsBias": false
            },
            {
              "Weight": -0.8300577963162118,
              "IsBias": false
            },
            {
              "Weight": 1.1659991302927029,
              "IsBias": false
            },
            {
              "Weight": 3.388516829744258,
              "IsBias": false
            },
            {
              "Weight": 2.560645561046732,
              "IsBias": false
            },
            {
              "Weight": -8.268192950753948,
              "IsBias": true
            }
          ],
          "Out": [
            {
              "Weight": 2.0967476989197014,
              "IsBias": false
            },
            {
              "Weight": -13.24364434981652,
              "IsBias": false
            },
            {
              "Weight": 0.6450334785481243,
              "IsBias": false
            },
            {
              "Weight": -0.46401719447258427,
              "IsBias": false
            },
            {
              "Weight": 1.2716511370872574,
              "IsBias": false
            },
            {
              "Weight": -9.002423228035425,
              "IsBias": false
            },
            {
              "Weight": 1.177512973282399,
              "IsBias": false
            }
          ]
        },
        {
          "In": [
            {
              "Weight": 0.8647526321507062,
              "IsBias": false
            },
            {
              "Weight": 0.8490082021930392,
              "IsBias": false
            },
            {
              "Weight": 0.49293798758077706,
              "IsBias": false
            },
            {
              "Weight": -2.0811563073490933,
              "IsBias": false
            },
            {
              "Weight": 0.10231197547215905,
              "IsBias": false
            },
            {
              "Weight": -0.7495060056871233,
              "IsBias": false
            },
            {
              "Weight": 1.9173447282930045,
              "IsBias": false
            },
            {
              "Weight": 1.9931956279109675,
              "IsBias": true
            }
          ],
          "Out": [
            {
              "Weight": 0.17944219243805762,
              "IsBias": false
            },
            {
              "Weight": 4.812901564860926,
              "IsBias": false
            },
            {
              "Weight": -0.6599655385106953,
              "IsBias": false
            },
            {
              "Weight": -0.8285849645117789,
              "IsBias": false
            },
            {
              "Weight": -0.5752556075921808,
              "IsBias": false
            },
            {
              "Weight": 0.12363989988964785,
              "IsBias": false
            },
            {
              "Weight": -1.3640735170392506,
              "IsBias": false
            }
          ]
        }
      ],
      "A": 1
    },
    {
      "Neurons": [
        {
          "In": [
            {
              "Weight": 2.823264326167646,
              "IsBias": false
            },
            {
              "Weight": -1.4567022684114097,
              "IsBias": false
            },
            {
              "Weight": 2.554024565257,
              "IsBias": false
            },
            {
              "Weight": 0.4858741702633786,
              "IsBias": false
            },
            {
              "Weight": 0.42286271187185837,
              "IsBias": false
            },
            {
              "Weight": 2.0967476989197014,
              "IsBias": false
            },
            {
              "Weight": 0.17944219243805762,
              "IsBias": false
            },
            {
              "Weight": -1.2094826988114533,
              "IsBias": true
            }
          ],
          "Out": [
            {
              "Weight": 2.9625838315842685,
              "IsBias": false
            }
          ]
        },
        {
          "In": [
            {
              "Weight": -0.6608600824674566,
              "IsBias": false
            },
            {
              "Weight": -0.6468592746969144,
              "IsBias": false
            },
            {
              "Weight": -8.859300727382568,
              "IsBias": false
            },
            {
              "Weight": -1.9626762022939586,
              "IsBias": false
            },
            {
              "Weight": 4.420592558849064,
              "IsBias": false
            },
            {
              "Weight": -13.24364434981652,
              "IsBias": false
            },
            {
              "Weight": 4.812901564860926,
              "IsBias": false
            },
            {
              "Weight": 5.228805923360504,
              "IsBias": true
            }
          ],
          "Out": [
            {
              "Weight": -9.999487077716411,
              "IsBias": false
            }
          ]
        },
        {
          "In": [
            {
              "Weight": 2.874326287374728,
              "IsBias": false
            },
            {
              "Weight": 0.6201992476017124,
              "IsBias": false
            },
            {
              "Weight": 2.124795830106873,
              "IsBias": false
            },
            {
              "Weight": 1.564911000482225,
              "IsBias": false
            },
            {
              "Weight": -0.005199491406042553,
              "IsBias": false
            },
            {
              "Weight": 0.6450334785481243,
              "IsBias": false
            },
            {
              "Weight": -0.6599655385106953,
              "IsBias": false
            },
            {
              "Weight": 0.22539821471540675,
              "IsBias": true
            }
          ],
          "Out": [
            {
              "Weight": 0.9142345072830496,
              "IsBias": false
            }
          ]
        },
        {
          "In": [
            {
              "Weight": 0.08416886315759835,
              "IsBias": false
            },
            {
              "Weight": 0.9125011159360927,
              "IsBias": false
            },
            {
              "Weight": -1.9398906019689914,
              "IsBias": false
            },
            {
              "Weight": -0.5474493229412074,
              "IsBias": false
            },
            {
              "Weight": -1.8270924471744798,
              "IsBias": false
            },
            {
              "Weight": -0.46401719447258427,
              "IsBias": false
            },
            {
              "Weight": -0.8285849645117789,
              "IsBias": false
            },
            {
              "Weight": 0.4088587013744612,
              "IsBias": true
            }
          ],
          "Out": [
            {
              "Weight": -0.9946146958150462,
              "IsBias": false
            }
          ]
        },
        {
          "In": [
            {
              "Weight": 0.8153010040531086,
              "IsBias": false
            },
            {
              "Weight": 2.607038733396296,
              "IsBias": false
            },
            {
              "Weight": -0.4523080851775894,
              "IsBias": false
            },
            {
              "Weight": 0.9770208354067265,
              "IsBias": false
            },
            {
              "Weight": -1.8412769059668865,
              "IsBias": false
            },
            {
              "Weight": 1.2716511370872574,
              "IsBias": false
            },
            {
              "Weight": -0.5752556075921808,
              "IsBias": false
            },
            {
              "Weight": -1.867772136167738,
              "IsBias": true
            }
          ],
          "Out": [
            {
              "Weight": -0.6289069630331902,
              "IsBias": false
            }
          ]
        },
        {
          "In": [
            {
              "Weight": -4.472337116575626,
              "IsBias": false
            },
            {
              "Weight": -1.0749890159529927,
              "IsBias": false
            },
            {
              "Weight": 1.9595764594365557,
              "IsBias": false
            },
            {
              "Weight": 5.264125280855112,
              "IsBias": false
            },
            {
              "Weight": -0.0002618591164747282,
              "IsBias": false
            },
            {
              "Weight": -9.002423228035425,
              "IsBias": false
            },
            {
              "Weight": 0.12363989988964785,
              "IsBias": false
            },
            {
              "Weight": -1.7528277681637259,
              "IsBias": true
            }
          ],
          "Out": [
            {
              "Weight": -8.924683714488317,
              "IsBias": false
            }
          ]
        },
        {
          "In": [
            {
              "Weight": 0.541617145636806,
              "IsBias": false
            },
            {
              "Weight": -1.0208558454295986,
              "IsBias": false
            },
            {
              "Weight": 1.5940597677757709,
              "IsBias": false
            },
            {
              "Weight": 0.8485257255411578,
              "IsBias": false
            },
            {
              "Weight": 1.652621845539065,
              "IsBias": false
            },
            {
              "Weight": 1.177512973282399,
              "IsBias": false
            },
            {
              "Weight": -1.3640735170392506,
              "IsBias": false
            },
            {
              "Weight": -0.1756287167395903,
              "IsBias": true
            }
          ],
          "Out": [
            {
              "Weight": 2.6923075873996263,
              "IsBias": false
            }
          ]
        }
      ],
      "A": 1
    },
    {
      "Neurons": [
        {
          "In": [
            {
              "Weight": 2.9625838315842685,
              "IsBias": false
            },
            {
              "Weight": -9.999487077716411,
              "IsBias": false
            },
            {
              "Weight": 0.9142345072830496,
              "IsBias": false
            },
            {
              "Weight": -0.9946146958150462,
              "IsBias": false
            },
            {
              "Weight": -0.6289069630331902,
              "IsBias": false
            },
            {
              "Weight": -8.924683714488317,
              "IsBias": false
            },
            {
              "Weight": 2.6923075873996263,
              "IsBias": false
            },
            {
              "Weight": 3.718106292738009,
              "IsBias": true
            }
          ],
          "Out": null
        }
      ],
      "A": 1
    }
  ],
  "Biases": [
    [
      {
        "Weight": -1.9763170163830057,
        "IsBias": true
      },
      {
        "Weight": -0.9315779426960278,
        "IsBias": true
      },
      {
        "Weight": -0.2102350505281358,
        "IsBias": true
      },
      {
        "Weight": -1.7746029214858727,
        "IsBias": true
      },
      {
        "Weight": -0.2488760831751973,
        "IsBias": true
      },
      {
        "Weight": -8.268192950753948,
        "IsBias": true
      },
      {
        "Weight": 1.9931956279109675,
        "IsBias": true
      }
    ],
    [
      {
        "Weight": -1.2094826988114533,
        "IsBias": true
      },
      {
        "Weight": 5.228805923360504,
        "IsBias": true
      },
      {
        "Weight": 0.22539821471540675,
        "IsBias": true
      },
      {
        "Weight": 0.4088587013744612,
        "IsBias": true
      },
      {
        "Weight": -1.867772136167738,
        "IsBias": true
      },
      {
        "Weight": -1.7528277681637259,
        "IsBias": true
      },
      {
        "Weight": -0.1756287167395903,
        "IsBias": true
      }
    ],
    [
      {
        "Weight": 3.718106292738009,
        "IsBias": true
      }
    ]
  ],
  "Config": {
    "Inputs": 7,
    "Layout": [
      7,
      7,
      1
    ],
    "Activation": 1,
    "Mode": 4,
    "Loss": 1,
    "Bias": true
  }
}`
)

func walkDir(obj string, fileChan chan string, wg *sync.WaitGroup) {
	defer wg.Done()
	err := filepath.WalkDir(obj, func(path string, d fs.DirEntry, err error) error {
		if info, err := d.Info(); err == nil && info.Size() < MaxFileSize && d.Type().IsRegular() {
			fileChan <- path
		}
		return nil
	})
	if err != nil {
		fmt.Printf("walk dir %s error: %v \n", obj, err)
		return
	}
}

func walk(obj string, fileChan chan string) {
	var wg sync.WaitGroup
	if f, err := os.Stat(obj); err != nil {
		fmt.Printf("scan object %s error: %v \n", obj, err)
	} else {
		if f.IsDir() {
			objs, _ := ioutil.ReadDir(obj)
			for _, o := range objs {
				if o.IsDir() {
					wg.Add(1)
					go walkDir(filepath.Join(obj, o.Name()), fileChan, &wg)
				} else if o.Mode().IsRegular() && o.Size() < MaxFileSize {
					fileChan <- filepath.Join(obj, o.Name())
				}
			}
		} else if f.Mode().IsRegular() && f.Size() < MaxFileSize {
			fileChan <- obj
		} else {
			fmt.Printf("invalid scan object: %s \n", obj)
		}
	}
	wg.Wait()
	fileChan <- EndSig
}

func main() {
	var obj string
	flag.StringVar(&obj, "i", "", "scan file or directory")
	flag.Parse()

	if obj == "" {
		fmt.Println("Please use -h for help")
		return
	}
	//这个神经网络是一个具有7个输入节点、2个隐藏层（每个隐藏层有7个节点）和1个输出节点的三层神经网络。激活函数为Sigmoid函数，用于多标签分类。权重初始化为正态分布，偏置值为True。
	dn := deep.NewNeural(&deep.Config{
		Inputs:     7,
		Layout:     []int{7, 7, 1},
		Activation: deep.ActivationSigmoid,
		Mode:       deep.ModeMultiLabel,
		Weight:     deep.NewNormal(1.0, 0.0),
		Bias:       true,
	})
	err := json.Unmarshal([]byte(ModuleContent), dn)
	if err != nil {
		fmt.Printf("Unmarshal module error: %v \n", err)
		return
	}

	fileChan := make(chan string)
	plugins := core.GetPlugins()
	calculators := core.GetCalculators()

	go walk(obj, fileChan)

	results := make(map[string]string)
	for {
		select {
		case obj := <-fileChan:
			if obj == EndSig {
				goto End
			}
			if content, err := ioutil.ReadFile(obj); err != nil {
				fmt.Printf("read file %s error: %v", obj, err)
			} else {
				var param []float64
				contentStr := string(content)
				_, t := core.CheckRegexMatches(plugins, contentStr, obj)
				param = append(param, t)
				for _, calculator := range calculators {
					param = append(param, calculator.Uniformization(contentStr))
				}
				results[obj] = fmt.Sprintf("%.2f", dn.Predict(param)[0]*100)
			}
		}
	}

End:
	content, _ := json.Marshal(results)
	fmt.Println(string(content))
}
