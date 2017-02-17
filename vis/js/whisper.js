Array.prototype.max = function() {
    return Math.max.apply(null, this);
};

Array.prototype.min = function() {
    return Math.min.apply(null, this);
};

function resetMaxDepth() {
    // find out the style of the current graph
    var style = $('.vis-style-group .btn-primary').attr('data');
    // find out the depth of the current graph
    var depth = $('div#' + style + "-chart").data('depth');
    // find out the current maxDepth of the graph
    var maxD = $('div#' + style + "-chart").data('maxD');

    // if depth is three, then max depth cannot be less than three
    $('.max-depth-group button').each(function() {
        console.log($(this).attr('data'), depth, maxD);
        if ($(this).attr('data') > depth) {
            $(this).removeClass('disabled');
        } else {
            $(this).addClass('disabled');
        }
        if ($(this).attr('data') == maxD) {
            $(this).addClass('btn-primary');
        } else {
            $(this).removeClass('btn-primary');

        }
    });
}

function maxDepth(maxD, style) {
    var style = typeof style !== 'undefined' ? style : $('.vis-style-group .btn-primary').attr('data');
    console.log(style);
    var svg;
    $('div#' + style + "-chart").data('maxD', maxD);

    if (style == 'packed') {

        svg = d3.select("#packed-chart svg");

        var circle = svg
            .selectAll("circle")
            .style("visibility", function(d) {
                return d.depth > maxD ? "hidden" : "visible";
            });
        colorPacked();
    }
    if (style == 'icicle') {

        svg = d3.select("#icicle-chart svg");
        var rect = svg.selectAll("rect")
            .style("visibility", function(d) {
                return d.depth > maxD ? "hidden" : "visible";
            });

    }

    if (style == 'sunburst') {

        svg = d3.select("#sunburst-chart svg");
        var text = svg.selectAll("path")
            .style("visibility", function(d) {
                return d.depth > maxD ? "hidden" : "visible";
            });
    }

}

function colorPacked(style) {
    var style = typeof style !== 'undefined' ? style : $('.color-entry-group .btn-primary').attr('data');
    var svg = d3.select("#packed-chart svg");
    var depth = $('#packed-chart').data('depth');

    // the palette used for encoding scores
    var color = d3.scale.linear()
        .domain([0, 0.2, 0.4, 0.6, 1])
        .range(["#637E84", "#92B4A5", "#EDC224", "#F9AC21", "#ED4E25"]);

    var colorG = d3.scale.linear()
        .domain([-1, 5])
        .range(["hsl(152,90%,90%)", "hsl(228,60%,20%)"])
        .interpolate(d3.interpolateHcl);

    var circle = svg
        .selectAll("circle")
        .style("fill", function(d) {
            var maxD = $('#packed-chart').data('maxD');
            var gini = d.info ? d.info.gini ? d.info.gini : 0 : 0;
            var modu = d.info ? d.info.sweetspot ? d.info.sweetspot : 0 : 0;
            modu = (modu * 9 < 1) ? modu * 9 : 1;
            var dist = d.info ? d.info.intraDistance ? d.info.intraDistance / d.info.interDistance : 1 : 1;
            dist = 1 - dist;
            // console.log(modu, color(modu));
            if (style == 'default' || d.depth != (depth + 1)) {
                return (d.children && d.depth != maxD) ? colorG(d.depth) : 'white';
            } else if (style == 'modu') {
                return (d.info && d.info.sweetspot && d.children) ? d3.rgb(color(modu)).darker(-1 + d.depth) : 'white';
            }
            return d3.rgb(color(dist)).darker(-1 + d.depth);
        });
}




function colorIcicle(style, transition) {
    transition = typeof transition !== 'undefined' ? transition : 0;
    // get the current depth
    var depth = $('#icicle-chart').data('depth');
    var colorG = d3.scale.linear()
        .domain([-1, 5])
        .range(["hsl(152,90%,90%)", "hsl(228,60%,20%)"])
        .interpolate(d3.interpolateHcl);

    var svg = d3.select("#icicle-chart svg");

    // var colorG = d3.scale.category20c();
    // the palette used for encoding scores
    var color = d3.scale.linear()
        .domain([0, 0.2, 0.4, 0.6, 1])
        .range(["#637E84", "#92B4A5", "#EDC224", "#F9AC21", "#ED4E25"]);

    var rect = svg.selectAll("rect")
        .transition()
        .delay(transition)
        .attr("fill", function(d) {
            var info = (d.children ? d : d.parent).info;
            var modu = d.info ? d.info.sweetspot ? d.info.sweetspot : 0 : 0;
            modu = (modu * 9 < 1) ? modu * 9 : 1;
            var dist = d.info ? d.info.intraDistance ? d.info.intraDistance / d.info.interDistance : 1 : 1;
            dist = 1 - dist;
            if (style == 'default' || d.depth != (depth + 1)) {
                return colorG(d.depth);
            } else if (style == 'modu') {
                return d3.rgb(color(modu));
            }
            return d3.rgb(color(dist));
        });
}

function colorSunburst(style, transition) {
    transition = typeof transition !== 'undefined' ? transition : 0;
    // get the current depth
    var depth = $('#sunburst-chart').data('depth');
    var colorG = d3.scale.linear()
        .domain([-1, 5])
        .range(["hsl(152,90%,90%)", "hsl(228,60%,20%)"])
        .interpolate(d3.interpolateHcl);

    var svg = d3.select("#sunburst-chart svg");

    // var colorG = d3.scale.category20c();
    // the palette used for encoding scores
    var color = d3.scale.linear()
        .domain([0, 0.2, 0.4, 0.6, 1])
        .range(["#637E84", "#92B4A5", "#EDC224", "#F9AC21", "#ED4E25"]);

    var text = svg.selectAll("path")
        .transition()
        .delay(transition)
        .style("fill", function(d) {
            // console.log(d.depth, depth);
            var info = (d.children ? d : d.parent).info;
            var modu = d.info ? d.info.sweetspot ? d.info.sweetspot : 0 : 0;
            modu = (modu * 9 < 1) ? modu * 9 : 1;
            var dist = d.info ? d.info.intraDistance ? d.info.intraDistance / d.info.interDistance : 1 : 1;
            dist = 1 - dist;

            if (style == 'default' || d.depth != (depth + 1)) {
                return colorG(d.depth);
            } else if (style == 'modu') {
                return d3.rgb(color(modu));
            }
            return d3.rgb(color(dist));
        });
}

function fillTreemap(d) {
    d = d.noChild ? d : d.parent;

    var colorG = d3.scale.linear()
        .domain([-1, 5])
        .range(["hsl(152,90%,90%)", "hsl(228,60%,20%)"])
        .interpolate(d3.interpolateHcl);

    // the palette used for encoding scores
    var color = d3.scale.linear()
        .domain([0, 0.2, 0.4, 0.6, 1])
        .range(["#637E84", "#92B4A5", "#EDC224", "#F9AC21", "#ED4E25"]);

    var style = $("#treemap-chart").data('color');
    var gini = d.info ? d.info.gini ? d.info.gini : 0 : 0;
    var modu = d.info ? d.info.sweetspot ? d.info.sweetspot : 0 : 0;
    modu = (modu * 9 < 1) ? modu * 9 : 1;
    var dist = d.info ? d.info.intraDistance ? d.info.intraDistance / d.info.interDistance : 1 : 1;
    dist = 1 - dist;
    var info = d.info ? d.info : null;
    if (style == 'default') {
        // console.log(d.depth);
        return colorG(d.depth);
    } else if (style == 'modu') {
        return color(modu);
    }
    return color(dist);

}

function colorTreemap(style) {

    // var colorG = d3.scale.category20c();

    $('#treemap-chart').data('color', style);
    var svg = d3.select("#treemap-chart svg");

    svg.selectAll(".child")
        .attr("fill", function(d) {
            return fillTreemap(d);
        })
}

$(function() {

    var urlParams;
    (window.onpopstate = function() {
        var match,
            pl = /\+/g, // Regex for replacing addition symbol with a space
            search = /([^&=]+)=?([^&/]*)/g,
            decode = function(s) {
                return decodeURIComponent(s.replace(pl, " "));
            },
            query = window.location.search.substring(1);

        urlParams = {};
        while (match = search.exec(query))
            urlParams[decode(match[1])] = decode(match[2]);
    })();

    $('#helper').scroll(function() {
        $('#helper-more').fadeOut('fast');
    });

    $('#display-helper').click(function() {
        console.log('triggered');
        $('#helper, #controller').toggle();
        $('#helper-more').toggleClass('hidden');
        $(this).find('span').toggleClass('glyphicon-resize-small');
        $(this).find('span').toggleClass('glyphicon-resize-full');
        $(this).toggleClass('bg');
        $('.main-container').toggleClass('expand');
    })

    $('body').append('<div class="detail"></div>');
    var detail = $('.detail');
    detail.hide();

    // the color for the to be labeled clusters
    // var color = d3.scale.linear()
    //     .domain([-2.5, 5])
    //     .range(["hsl(0,0%,90%)", "hsl(0,0%,40%)"])
    //     .interpolate(d3.interpolateHcl);

    // enable the controller of the visulization style
    $('.vis-style-group button').click(function() {
        $('div[id$=chart]').hide();
        $('div#' + $(this).attr('data') + "-chart").show();
        $(this).siblings().removeClass('btn-primary');
        $(this).addClass('btn-primary');
        resetMaxDepth();
        if ($(this).attr('data') == 'treemap') {
            $('.max-depth-group button').addClass('disabled');
        }

    });

    $('#controller-helper').click(function() {
        $(this).find('.glyphicon').toggleClass('hidden');
        $('#controller .panel-body').toggle();
        $('#helper').toggleClass('expand');
    })

    $('.color-entry-group button').click(function() {
        console.log('colro btn clicked');
        colorPacked($(this).attr('data'));
        colorIcicle($(this).attr('data'));
        colorSunburst($(this).attr('data'));
        colorTreemap($(this).attr('data'));
        $(this).siblings().removeClass('btn-primary');
        $(this).addClass('btn-primary');
    });

    $('.max-depth-group button').click(function() {
        if ($(this).hasClass('disabled')) { return false; }
        maxDepth($(this).attr('data'));
        $(this).siblings().removeClass('btn-primary');
        $(this).addClass('btn-primary');
    });


    var maxD = 5;
    $('div[id$=chart]').data('maxD', maxD);
    $('#treemap-chart').data('maxD', 10);

    jsonFile = urlParams.json;
    username = urlParams.user;
    labelLog = [];
    // $.post( "/send", { username: username, cid: 'started', label: 'null', file: jsonFile } );
    // labelLog[labelLog.length] = [Math.floor(Date.now() / 1000), 'started', 'null'];

    function renderCounter() {
        $('#counter').html('<b>Labeled custers:</b> ' + totalLabeled + '/' + totalToLabel);
        // if (totalLabeled == totalToLabel) {
        //     var submitBtn = $('<a class="btn btn-primary btn-sm">Submit</a>');
        //     $('#counter').append(submitBtn);
        //     submitBtn.click(function() {
        //         $.post("/full", {
        //                 username: username,
        //                 file: jsonFile,
        //                 data: JSON.stringify(labelLog)
        //             })
        //             .done(function() {
        //                 alert("Uploaded!");
        //             }).fail(function() {
        //                 alert("Error, please try again later.");
        //             });
        //     })
        // }
    }

    function addFigure(curClass, myData) {

        // add bar chart
        // var data = [4, 8, 15, 16, 23, 42];
        // var dataP = [16,5,4,3,2,1];
        var data = [];
        for (var i = 0; i < myData[0].length; i += 2) {
            data[data.length] = myData[0][i] + myData[0][i + 1];
        };
        var dataP = [];
        for (var i = 0; i < myData[0].length; i += 2) {
            dataP[dataP.length] = myData[1][i] + myData[1][i + 1];
        };
        // var data = myData[0];
        // var dataP = myData[1];

        var height = 20,
            barWidth = 6;

        var maxV = d3.max([d3.max(data), d3.max(dataP)]);
        // console.log(maxV);
        var y = d3.scale.linear()
            .domain([0, maxV])
            .range([0, height]);

        var y2 = d3.scale.linear()


        // console.log($(curClass));
        var chart = d3.select(curClass)
            .attr("height", height)
            .attr("width", barWidth * data.length);

        var bar = chart.selectAll("g")
            .data(data)
            .enter().append("g")
            .attr("transform", function(d, i) {
                return "translate(" + i * barWidth + ", 0)";
            });


        bar.append("rect")
            .data(dataP)
            .attr("y", function(d) {
                return height - y(d)
            })
            .attr("height", y)
            .attr("x", barWidth * 0.5 - 0.2)
            .attr("fill", "teal")
            .style("opacity", 0.9)
            .attr("width", barWidth * 0.5 - 0.2);

        bar.append("rect")
            .data(data)
            .attr("y", function(d) {
                return height - y(d)
            })
            .attr("height", y)
            .attr("fill", "red")
            .style("opacity", 0.9)
            .attr("width", barWidth * 0.5 - 0.2);
    }

    function renderPacked() {
        $('#packed-chart').data('depth', 0);

        var margin = 20,
            diameter = 800;

        var colorG = d3.scale.linear()
            .domain([-1, 5])
            .range(["hsl(152,90%,90%)", "hsl(228,60%,20%)"])
            .interpolate(d3.interpolateHcl);

        var pack = d3.layout.pack()
            .padding(2)
            .size([diameter - margin, diameter - margin])
            .value(function(d) {
                return d.size;
            })

        var svg = d3.select("#packed-chart").append("svg")
            .attr("width", diameter)
            .attr("height", diameter)
            .append("g")
            .attr("transform", "translate(" + diameter / 2 + "," + diameter / 2 + ")");

        totalToLabel = 0;
        totalLabeled = 0;

        d3.json(urlParams.json, function(error, root) {
            if (error) return console.error(error);
            var focus = root,
                nodes = pack.nodes(root),
                view;

            for (var i = nodes.length - 1; i >= 0; i--) {
                var node = nodes[i];
                if ('info' in node && 'tolabel' in node.info && node.info.tolabel) {
                    totalToLabel += 1;
                }
            };

            console.log(totalToLabel);
            renderCounter();
            var circle = svg
                .selectAll("circle")
                .data(nodes)
                .enter().append("circle")
                .attr("class", function(d) {
                    return d.parent ? d.children ? "node" : "node node--leaf" : "node node--root";
                })
                // .style("fill", function(d) { return d.children ? color(d.depth) : null; })
                .style("fill", function(d) {
                    return d.children ? colorG(d.depth) : null;
                })
                .on("dblclick", function(d) {
                    if ($(this).attr('class').search('node--leaf') >= 0 ||
                        d.depth == $('#packed-chart').data('maxD')) {
                        if (focus !== d.parent) {
                            zoom(d.parent), d3.event.stopPropagation();
                        }
                        return;
                    }
                    if (focus !== d) {
                        zoom(d), d3.event.stopPropagation();
                    }
                })
                // display information related to the given cluster, should be universal
                .on('click', function(d) {
                    displayDetail(d, text, this)
                });

            // initial coloring
            var text = svg.selectAll("text")
                .data(nodes)
                .enter().append("text")
                .attr("class", "label")
                .style("fill-opacity", function(d) {
                    return d.parent === root ? 1 : 0;
                })
                .style("display", function(d) {
                    return d.parent === root ? null : "none";
                })
                .text(function(d) {
                    return d.name;
                });

            var node = svg.selectAll("circle,text");

            // handles zooming functionalities, specific to packed circle

            d3.select("body")
                .style("background", "hsl(152,90%,90%)")
                .on("dblclick", function() {
                    // if the current visulization style is packed circle
                    if ($('.vis-style-group .btn-primary').attr('data') == 'packed') {
                        zoom(root);

                    }
                });

            zoomTo([root.x, root.y, root.r * 2 + margin]);

            function zoom(d) {
                $('#packed-chart').data('depth', d.depth);
                resetMaxDepth();

                var focus0 = focus;
                focus = d;

                var transition = svg.transition()
                    .duration(d3.event && d3.event.altKey ? 7500 : 750)
                    .tween("zoom", function(d) {
                        var i = d3.interpolateZoom(view, [focus.x, focus.y, focus.r * 2 + margin]);
                        return function(t) {
                            zoomTo(i(t));
                        };
                    });

                transition.selectAll("text")
                    .filter(function(d) {
                        // console.log(d);
                        return d.parent === focus || this.style.display === "block";
                    })
                    .style("fill-opacity", function(d) {
                        return d.parent === focus ? 1 : 0;
                    })
                    .each("start", function(d) {
                        if (d.parent === focus) this.style.display = "block";
                    })
                    .each("end", function(d) {
                        if (d.parent !== focus) this.style.display = "none";
                    });
                colorPacked($('.color-entry-group .btn-primary').attr('data'), 750);

                detail.hide();
            }

            function zoomTo(v) {
                var k = diameter / v[2];
                view = v;
                node.attr("transform", function(d) {
                    return "translate(" + (d.x - v[0]) * k + "," + (d.y - v[1]) * k + ")";
                });
                text.attr("transform", function(d) {
                    return "translate(" + (d.x - v[0]) * k + "," + (d.y - v[1] + d.r) * k + ")";
                });
                circle.attr("r", function(d) {
                    return d.r * k;
                });
            }
            colorPacked('default');

        });

        d3.select(self.frameElement).style("height", diameter + "px");

    }

    function renderSunburst() {
        $('#sunburst-chart').data('depth', 0);
        var width = 960,
            height = 700,
            radius = (Math.min(width, height) / 2) - 10;

        var formatNumber = d3.format(",d");

        var x = d3.scale.linear()
            .range([0, 2 * Math.PI]);

        var y = d3.scale.sqrt()
            .range([0, radius]);

        // var color = d3.scale.category20c();

        var partition = d3.layout.partition()
            .value(function(d) {
                return d.size;
            });

        var arc = d3.svg.arc()
            .startAngle(function(d) {
                return Math.max(0, Math.min(2 * Math.PI, x(d.x)));
            })
            .endAngle(function(d) {
                return Math.max(0, Math.min(2 * Math.PI, x(d.x + d.dx)));
            })
            .innerRadius(function(d) {
                return Math.max(0, y(d.y));
            })
            .outerRadius(function(d) {
                return Math.max(0, y(d.y + d.dy) - 1);
            });

        var svg = d3.select("#sunburst-chart").append("svg")
            .attr("width", width)
            .attr("height", height)
            .append("g")
            .attr("transform", "translate(" + width / 2 + "," + (height / 2) + ")");

        d3.json(jsonFile, function(error, root) {
            if (error) throw error;

            var text = svg.selectAll("path")
                .data(partition.nodes(root))
                .enter().append("path")
                .attr("d", arc)
                // .style("fill", function(d) {
                //     var info = (d.children ? d : d.parent).info;
                //     return color(info ? info.id : null);
                // })
                .on("dblclick", click)
                .on("click", function(d) {
                    displayDetail(d, text, this)
                })
                .append("title")
                .text(function(d) {
                    return d.name + "\n" + formatNumber(d.value);
                });
            colorSunburst('default');
        });

        function click(d) {
            $('#sunburst-chart').data('depth', d.depth);
            resetMaxDepth();

            svg.transition()
                .duration(750)
                .tween("scale", function() {
                    var xd = d3.interpolate(x.domain(), [d.x, d.x + d.dx]),
                        yd = d3.interpolate(y.domain(), [d.y, 1]),
                        yr = d3.interpolate(y.range(), [d.y ? 20 : 0, radius]);
                    return function(t) {
                        x.domain(xd(t));
                        y.domain(yd(t)).range(yr(t));
                    };
                })
                .selectAll("path")
                .attrTween("d", function(d) {
                    return function() {
                        return arc(d);
                    };
                });
            colorSunburst($('.color-entry-group .btn-primary').attr('data'), 750);

            detail.hide();

        }

        d3.select(self.frameElement).style("height", height + "px");
    }

    function displayDetail(d, text, theRec) {
        if (d.name == "root") {
            return;
        }

        function selectElement(ele) {
            // remove the class slected from the old selected cluster, should be universal
            // $('.selected').removeClass('selected');
            d3.selectAll(".selected")
                .classed("selected", false);
            if (ele.attr('class')) {
                ele.attr('class', ele.attr('class') + ' selected');
            } else {
                ele.attr('class', 'selected');
            }
        }
        selectElement($(theRec));

        var actionTime = {
            5: "1S",
            6: "1M",
            7: "1H",
            8: "1D",
            9: "1D+"
        };

        // console.log($(this).attr('class'));
        // $(this).addClass('selected');
        // console.log(d.name);

        // display cluster summary, should be universal
        detail.show();
        detail.html('');
        detail.prepend($('<div class="summary"><span><b>Cluster ID:</b> ' + d.info.id + '<b> | Number of Users:</b> ' + d.info.size + ' users</span></div>'));

        // handle add cluster label, should be unversal
        // clickLink = $('<a class="label-btn btn-comment btn btn-primary" href="javascript:;">add label</a>');
        // clickLink.click(function(event) {
        //     var label = prompt("label this cluster", "");
        //     if (label) {
        //         if (d.info.tolabel) {
        //             // we have lebeled a new cluster
        //             totalLabeled += 1;
        //             renderCounter();
        //         }
        //         // TODO: need to change this, propogate name change to all visulizations
        //         d.name = label;
        //         // $.post("/send", {
        //         //     username: username,
        //         //     cid: d.info.id,
        //         //     label: label,
        //         //     file: jsonFile
        //         // });
        //         labelLog[labelLog.length] = [Math.floor(Date.now() / 1000), d.info.id, label];

        //         text.text(function(d) {
        //             return d.name;
        //         });
        //         d.info.tolabel = false;
        //         // circle.style("fill", function(d) { 
        //         //   if ('info' in d && 'tolabel' in d.info && !d.info.tolabel){
        //         //     return d.children ? color(d.depth) : null;
        //         //   }else{
        //         //     return d.children ? colorG(d.depth) : null;
        //         //   }
        //         // });
        //         detail.hide();
        //         // TODO: this is packed circle specific, need to change that
        //         // zoom(d.parent);
        //     }
        //     event.stopPropagation();
        // });
        // console.log(d.info.sweetspot);
        // display the 
        // detail.append($('<p>'+JSON.stringify(d.info, null, 2)+"</p>")); 
        // detail.find('.summary').append(clickLink);
        // if (d.name) {
        //     clickLink.attr('data-toggle', 'tooltip');
        //     clickLink.attr('title', d.name);
        //     clickLink.tooltip();
        // }
        // detail.append($('<h3>Features</h3>'));
        // display action patterns, should be universal
        features = $('<table class="table table-condensed features"></table>');
        // features.append($('<tr><th>Rank</th><th>Action Pattern</th><th>Frequency<br>Distribution</th><th>Average<br>Frequency<br>(In : Out)</th><th>Score</th></tr>'));
        features.append($('<tr><th>Rank</th><th>Action Pattern</th><th>Frequency<br>Distribution</th><th>Score</th></tr>'));
        detail.append(features);

        // get a list of all scores
        // d.info.exclusions[i][3]
        var scores = [];
        $.map(d.info.exclusions, function(x) {
            scores.push(x[3])
        });
        // console.log(scores, scores.min(), scores.max());
        var scoreColor = d3.scale.linear()
            .domain([0, scores.max()])
            .range(["#6ecddd", "#f98909"])
            .interpolate(d3.interpolateHcl);
        // console.log(scoreColor(scores[1]));


        for (var i = 0; i < d.info.exclusions.length; i++) {
            // console.log(d.info.exclusions[i]);
            var feature = d.info.exclusions[i][1].split(", ");
            // console.log(feature);
            var display = $('<td class="feature"></td>');
            for (var j = 0; j < feature.length; j++) {
                var action = feature[j];
                if (action.length == 0) {
                    continue;
                }
                if (j % 2 == 0) {
                    // add a action item
                    display.append($('<span class="label label-primary action action-' + action.replace(/\s+/g, '') + '">' + action + '</span>'));
                } else {
                    display.append($('<span class="timegap label label-success">' + actionTime[action] + '</span>'));
                }
            };
            var row = $('<tr></tr>');
            row.append($('<td>' + (i + 1) + '</td>'));
            row.append(display);
            row.append('<td><svg class="chart' + i + '"></svg></td>');
            features.append(row);
            // console.log(row.html());
            addFigure('.chart' + i, d.info.exclusions[i][2]);
            // row.append('<td class="no-wrap"><span class="local-avg">'+d.info.exclusions[i][4][0].toFixed(2)
            //   +'</span> : <span class="global-avg">'+d.info.exclusions[i][4][1].toFixed(2)+'</span></td>');

            // print chi-square score
            var curScore = d.info.exclusions[i][3];
            // console.log(scoreColor(curScore));
            scoreText = $('<span class="label">' + curScore.toFixed(2) + '</span>')
                .css('background-color', scoreColor(curScore));
            row.append($('<td></td>').append(scoreText));
        };

        if (d.name) {
            detail.append('<div><b>Label:</b> ' + d.name + ' </div>');
        }
        var closeBt = $('<a class="btn btn-close" href="javascript:;">x</a>');
        detail.append(closeBt);
        closeBt.click(function() {
            detail.hide();
        });
        detail.css({
            'top': (d3.event.pageY + 4) + 'px',
            'left': (d3.event.pageX + 4) + 'px'
        });
    }

    function renderIcicle() {
        $('#icicle-chart').data('depth', 0);
        var width = 780,
            height = 600;

        var x = d3.scale.linear()
            .range([0, width]);

        var y = d3.scale.linear()
            .range([0, height]);


        var partition = d3.layout.partition()
            .children(function(d) {
                return isNaN(d.children) ? d.children : null;
            })
            .value(function(d) {
                return d.size;
            });

        var svgI = d3.select("#icicle-chart").append("svg")
            .attr("width", width)
            .attr("height", height);
        var rect;
        var text;
        d3.json(jsonFile, function(error, root) {
            if (error) throw error;

            var parts = partition(root);
            // $(parts).each(function(d, v){v.info = labels[v.key]});

            rect = svgI.selectAll("rect")
                .data(parts)
                .enter().append("rect")
                .attr("x", function(d) {
                    // console.log(d);
                    return x(d.x);
                })
                .attr("y", function(d) {
                    return y(d.y);
                })
                .attr("width", function(d) {
                    return x(d.dx) > 2 ? x(d.dx) - 2 : 0;
                })
                .attr("height", function(d) {
                    // console.log(d.dy);
                    return y(d.dy) > 2 ? y(d.dy) - 2 : 0;
                })
                .on("click", function(d) {
                    displayDetail(d, text, this)
                })
                .on("dblclick", clicked);

            text = svgI.selectAll(".label")
                .data(parts.filter(function(d) {
                    return x(d.dx) > 6;
                }))
                .enter().append("text")
                .attr("class", "label")
                .attr("dy", ".35em")
                .attr("transform", function(d) {
                    return "translate(" + x(d.x + d.dx / 2) + "," + y(d.y + d.dy / 2) + ")rotate(90)";
                })
                .text(function(d) {
                    return d.name;
                });
            colorIcicle("default");
        });

        function clicked(d) {
            // console.log(d.info);
            x.domain([d.x, d.x + d.dx]);
            y.domain([d.y, 1]).range([d.y ? 20 : 0, height]);
            $('#icicle-chart').data('depth', d.depth);
            resetMaxDepth();

            rect.transition()
                .duration(750)
                .attr("x", function(d) {
                    return x(d.x);
                })
                .attr("y", function(d) {
                    return y(d.y);
                })
                .attr("width", function(d) {
                    var val = x(d.x + d.dx) - x(d.x);
                    return val > 2 ? val - 2 : 0;
                })
                .attr("height", function(d) {
                    var val = y(d.y + d.dy) - y(d.y);
                    return val > 2 ? val - 2 : 0;
                });

            text.transition()
                .duration(750)
                .attr("dy", ".35em")
                .attr("transform", function(d) {
                    return "translate(" + x(d.x + d.dx / 2) + "," + y(d.y + d.dy / 2) + ")rotate(90)";
                });
            $('#icicle-chart').data('depth', d.depth);

            colorIcicle($('.color-entry-group .btn-primary').attr('data'), 750);
            detail.hide();

        }

    }

    function renderTreemap() {
        $('#treemap-chart').data('depth', 0);

        var margin = {
                top: 20,
                right: 0,
                bottom: 0,
                left: 0
            },
            width = 760,
            height = 560 - margin.top - margin.bottom,
            formatNumber = d3.format(",d"),
            transitioning;

        var x = d3.scale.linear()
            .domain([0, width])
            .range([0, width]);

        var y = d3.scale.linear()
            .domain([0, height])
            .range([0, height]);


        var treemap = d3.layout.treemap()
            .children(function(d, depth) {
                return depth ? null : d._children;
            })
            .sort(function(a, b) {
                return a.value - b.value;
            })
            .ratio(height / width * 0.5 * (1 + Math.sqrt(5)))
            .round(false);

        var svg = d3.select("#treemap-chart").append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.bottom + margin.top)
            .style("margin-left", -margin.left + "px")
            .style("margin.right", -margin.right + "px")
            .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
            .style("shape-rendering", "crispEdges");

        var grandparent = svg.append("g")
            .attr("class", "grandparent");

        grandparent.append("rect")
            .attr("y", -margin.top)
            .attr("width", width)
            .attr("height", margin.top);

        grandparent.append("text")
            .attr("x", 6)
            .attr("y", 6 - margin.top)
            .attr("dy", ".75em");

        $("#treemap-chart").data('color', 'default');

        d3.json(jsonFile, function(root) {
            initialize(root);
            accumulate(root);
            layout(root);
            display(root);

            function initialize(root) {
                root.x = root.y = 0;
                root.dx = width;
                root.dy = height;
                root.depth = 0;
            }

            // Aggregate the values for internal nodes. This is normally done by the
            // treemap layout, but not here because of our custom implementation.
            // We also take a snapshot of the original children (_children) to avoid
            // the children being overwritten when when layout is computed.
            function accumulate(d) {
                return (d._children = d.children) ? d.value = d.children.reduce(function(p, v) {
                    return p + accumulate(v);
                }, 0) : d.value = d.size;
            }

            // Compute the treemap layout recursively such that each group of siblings
            // uses the same size (1×1) rather than the dimensions of the parent cell.
            // This optimizes the layout for the current zoom state. Note that a wrapper
            // object is created for the parent node for each group of siblings so that
            // the parent’s dimensions are not discarded as we recurse. Since each group
            // of sibling was laid out in 1×1, we must rescale to fit using absolute
            // coordinates. This lets us use a viewport to zoom.
            function layout(d) {
                if (d._children) {
                    treemap.nodes({
                        _children: d._children
                    });
                    d._children.forEach(function(c) {
                        c.x = d.x + c.x * d.dx;
                        c.y = d.y + c.y * d.dy;
                        c.dx *= d.dx;
                        c.dy *= d.dy;
                        c.parent = d;
                        c.depth = d.depth + 1;
                        layout(c);
                    });
                }
            }


            function display(d) {
                grandparent
                    .datum(d.parent)
                    .on("click", transition)
                    .select("text")
                    .text(name(d));

                var g1 = svg.insert("g", ".grandparent")
                    .datum(d)
                    .attr("class", "depth");

                var g = g1.selectAll("g")
                    .data(d._children)
                    .enter().append("g");

                g.on("click", function(d) {
                        displayDetail(d, labels, this)
                    })
                    .classed("children", true)
                    .filter(function(d) {
                        return d._children;
                    })
                    .on('mouseover', function(d) {
                        // console.log('triggered');
                        d3.select(this).style({
                            'fill-opacity': '0.5'
                        });
                    })
                    .on('mouseout', function(d) {
                        // console.log('triggered');
                        d3.select(this).style({
                            'fill-opacity': '1'
                        });
                    })
                    .on("dblclick", transition);


                g.selectAll(".child")
                    .data(function(d) {
                        if (!d._children) {
                            d = $.extend({}, d);
                            d.noChild = true;
                        }
                        return d._children || [d];
                    })
                    .enter().append("rect")
                    .attr("class", "child")
                    .call(rect)
                    .attr("fill", function(d) {
                        return fillTreemap(d);
                    });

                g.append("rect")
                    .attr("class", "parent")
                    .call(rect)
                    .attr("fill", function(d) {
                        // return fillTreemap(d, 'modu');
                    })
                    .attr("fill-opacity", 0)
                    .append("title")
                    .text(function(d) {
                        return formatNumber(d.value);
                    });

                var labels = g.append("text")
                    .attr("dy", ".75em")
                    .text(function(d) {
                        return d.name;
                    })
                    .call(text);

                function transition(d) {
                    // console.log(d);

                    if (transitioning || !d) return;
                    $('#treemap-chart').data('depth', d.depth);

                    transitioning = true;

                    var g2 = display(d),
                        t1 = g1.transition().duration(750),
                        t2 = g2.transition().duration(750);

                    // Update the domain only after entering new elements.
                    x.domain([d.x, d.x + d.dx]);
                    y.domain([d.y, d.y + d.dy]);

                    // Enable anti-aliasing during the transition.
                    svg.style("shape-rendering", null);

                    // Draw child nodes on top of parent nodes.
                    svg.selectAll(".depth").sort(function(a, b) {
                        return a.depth - b.depth;
                    });

                    // Fade-in entering text.
                    g2.selectAll("text").style("fill-opacity", 0);

                    // Transition to the new view.
                    t1.selectAll("text").call(text).style("fill-opacity", 0);
                    t2.selectAll("text").call(text).style("fill-opacity", 1);
                    t1.selectAll("rect").call(rect).style("fill-opacity", 0);
                    t2.selectAll(".parent").call(rect).style("fill-opacity", 0);
                    t2.selectAll(".child").call(rect).style("fill-opacity", 1);

                    // Remove the old node when the transition is finished.
                    t1.remove().each("end", function() {
                        svg.style("shape-rendering", "crispEdges");
                        transitioning = false;
                    });
                }
                detail.hide();
                return g;
            }

            function text(text) {
                text.attr("x", function(d) {
                        return x(d.x) + 6;
                    })
                    .attr("y", function(d) {
                        return y(d.y) + 6;
                    });
            }

            function rect(rect) {
                rect.attr("x", function(d) {
                        return x(d.x);
                    })
                    .attr("y", function(d) {
                        return y(d.y);
                    })
                    .attr("width", function(d) {
                        return x(d.x + d.dx) - x(d.x);
                    })
                    .attr("height", function(d) {
                        return y(d.y + d.dy) - y(d.y);
                    });
            }

            function name(d) {
                return d.parent ? name(d.parent) + "." + d.name : d.name;
            }
        });
    }

    // render packed circle
    renderPacked();
    renderIcicle();
    renderSunburst();
    renderTreemap();
});